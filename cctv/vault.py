"""Encrypted face vault: tamper-evident storage for biometric templates.

Why a vault?
------------
The old system kept family faces as plain JPGs in ``family/<Name>/`` and
unknown faces only as snapshot images. Anyone with disk access could
copy, edit, or delete them — and biometric data cannot be "re-issued"
like a password. This module fixes that with three layers:

1. **Encryption** — every row (name + 128-d face encoding) is sealed in
   an AES-256-GCM authenticated blob (cctv/crypto.py). A stolen vault.db
   is useless without the key.

2. **Two separate tables** —
     * ``family_faces``  : the household's enrolled templates.
     * ``unknown_faces`` : encodings of strangers the camera saw, so
       repeat intruders are recognised and flagged faster.
   Both are encrypted with the same cipher but never mixed.

3. **Tamper evidence (HMAC chain)** — every insert/update/delete is
   recorded in an append-only ``audit_chain`` table where each entry's
   HMAC covers the previous entry's HMAC (a hash chain). Deleting or
   editing a row breaks the chain, and the next integrity scan reports
   exactly which table and row id were affected. Deletions are also
   *tombstoned* (soft-delete) so a hacker cannot silently erase a face.

Public API (used by faces.py, register.py, enroll.py, main.py)
----------------------------------------------------------------
    FaceVault(path, cipher)          open/create the vault
    vault.add_family_face(name, enc) encrypt + store one family template
    vault.load_family()               decrypt all family encodings + names
    vault.record_unknown(enc, meta)  store an intruder template (deduped)
    vault.load_unknown()             decrypt unknown templates
    vault.delete_family_person(name) tombstone + audit (family removal)
    vault.verify_integrity()         full HMAC chain + row-level check
    vault.enforce_unknown_retention() age out old intruder records
"""

import json
import os
import sqlite3
import threading

from datetime import datetime, timedelta

import numpy as np

from config import (
    VAULT_UNKNOWN_RETENTION_DAYS,
    VAULT_MAX_UNKNOWN_PER_PERSON,
    VAULT_UNKNOWN_MATCH_DISTANCE,
    LOG_DIR,
)

from cctv.crypto import VaultCipher, VaultTamperError

# Default vault location: alongside the events DB in logs/.
VAULT_DB = os.path.join(LOG_DIR, "vault.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS family_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sealed BLOB NOT NULL,          -- AES-GCM(name + encoding)
    sealed_hmac BLOB NOT NULL,     -- row-level tamper seal
    created_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0   -- tombstone (soft delete)
);

CREATE TABLE IF NOT EXISTS unknown_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sealed BLOB NOT NULL,          -- AES-GCM(encoding + metadata)
    sealed_hmac BLOB NOT NULL,
    created_at TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    sightings INTEGER NOT NULL DEFAULT 1,
    deleted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_chain (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    action TEXT NOT NULL,          -- INSERT / UPDATE / DELETE
    prev_hmac BLOB NOT NULL,       -- chain link to the previous entry
    entry_hmac BLOB NOT NULL       -- HMAC over this entry's fields
);

CREATE INDEX IF NOT EXISTS idx_family_deleted
    ON family_faces(deleted);
CREATE INDEX IF NOT EXISTS idx_unknown_deleted
    ON unknown_faces(deleted);
"""


class VaultIntegrityError(RuntimeError):
    """Raised when the audit chain or a row seal is broken."""


class FaceVault:
    """SQLite-backed, AES-encrypted, tamper-evident face store."""

    def __init__(self, path: str = VAULT_DB, cipher: VaultCipher | None = None):
        self.path = path
        self.cipher = cipher or VaultCipher.from_config()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _seal_record(self, payload: dict) -> tuple[bytes, bytes]:
        """Encrypt a JSON payload; return (sealed_blob, row_hmac)."""
        blob = self.cipher.seal(json.dumps(payload).encode("utf-8"))
        mac = self.cipher.hmac(blob)
        return blob, mac

    def _unseal_record(self, sealed: bytes, mac: bytes) -> dict:
        """Verify the row HMAC, then decrypt. Raises on tampering."""
        if not self.cipher.hmac_verify(sealed, mac):
            raise VaultIntegrityError(
                "Row HMAC mismatch — record was modified outside the vault."
            )
        return json.loads(self.cipher.unseal(sealed).decode("utf-8"))

    def _audit(self, table: str, row_id: int, action: str) -> None:
        """Append a chained audit entry: HMAC(prev) links entries."""
        cur = self._conn.execute(
            "SELECT entry_hmac FROM audit_chain ORDER BY seq DESC LIMIT 1"
        )
        prev = cur.fetchone()
        prev_hmac = prev[0] if prev else b"\x00" * 32
        entry_hmac = self.cipher.hmac(
            f"{table}|{row_id}|{action}|".encode("utf-8")
            + prev_hmac
        )
        self._conn.execute(
            """
            INSERT INTO audit_chain
            (table_name, row_id, action, prev_hmac, entry_hmac)
            VALUES (?, ?, ?, ?, ?)
            """,
            (table, row_id, action, prev_hmac, entry_hmac),
        )

    # ------------------------------------------------------------------
    # Family faces
    # ------------------------------------------------------------------

    def add_family_face(self, name: str, encoding) -> int:
        """Encrypt and store one family face template. Returns row id."""
        enc = np.asarray(encoding, dtype=np.float64).tolist()
        blob, mac = self._seal_record({"name": name, "encoding": enc})
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO family_faces (sealed, sealed_hmac, created_at)
                VALUES (?, ?, ?)
                """,
                (blob, mac, datetime.now().isoformat(timespec="seconds")),
            )
            row_id = int(cur.lastrowid)
            self._audit("family_faces", row_id, "INSERT")
            self._conn.commit()
        return row_id

    def load_family(self) -> tuple[list, list]:
        """Decrypt every live family template.

        Returns (encodings, names) exactly like the old
        load_family_database() so faces.py can use the vault unchanged.
        """
        encodings, names = [], []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT sealed, sealed_hmac FROM family_faces
                WHERE deleted = 0
                """
            ).fetchall()
        for sealed, mac in rows:
            try:
                rec = self._unseal_record(sealed, mac)
            except (VaultIntegrityError, VaultTamperError):
                # Skip tampered rows but keep the system running; the
                # startup integrity scan is the loud failure point.
                continue
            encodings.append(np.array(rec["encoding"], dtype=np.float64))
            names.append(rec["name"])
        return encodings, names

    def delete_family_person(self, name: str) -> int:
        """Tombstone every template of *name*; audit the deletion."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM family_faces WHERE deleted = 0"
            ).fetchall()
            removed = 0
            for (row_id,) in rows:
                sealed, mac = self._conn.execute(
                    "SELECT sealed, sealed_hmac FROM family_faces WHERE id = ?",
                    (row_id,),
                ).fetchone()
                try:
                    rec = self._unseal_record(sealed, mac)
                except (VaultIntegrityError, VaultTamperError):
                    continue
                if rec.get("name") == name:
                    self._conn.execute(
                        "UPDATE family_faces SET deleted = 1 WHERE id = ?",
                        (row_id,),
                    )
                    self._audit("family_faces", row_id, "DELETE")
                    removed += 1
            self._conn.commit()
        return removed

    # ------------------------------------------------------------------
    # Unknown (intruder) faces
    # ------------------------------------------------------------------

    def record_unknown(self, encoding, metadata: dict | None = None) -> int:
        """Store an intruder template, deduplicating repeat visitors.

        If the encoding matches a recent unknown record (distance ≤
        VAULT_UNKNOWN_MATCH_DISTANCE), that record's sighting counter
        and last_seen are bumped instead of inserting a duplicate.
        Returns the row id touched.
        """
        enc = np.asarray(encoding, dtype=np.float64)
        now_iso = datetime.now().isoformat(timespec="seconds")

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, sealed, sealed_hmac FROM unknown_faces
                WHERE deleted = 0
                """
            ).fetchall()

            for row_id, sealed, mac in rows:
                try:
                    rec = self._unseal_record(sealed, mac)
                except (VaultIntegrityError, VaultTamperError):
                    continue
                known = np.array(rec["encoding"], dtype=np.float64)
                if known.shape == enc.shape:
                    dist = float(np.linalg.norm(known - enc))
                    if dist <= VAULT_UNKNOWN_MATCH_DISTANCE:
                        rec["sightings"] = int(rec.get("sightings", 1)) + 1
                        rec["last_seen"] = now_iso
                        if metadata:
                            rec.setdefault("meta", {}).update(metadata)
                        blob, new_mac = self._seal_record(rec)
                        self._conn.execute(
                            """
                            UPDATE unknown_faces
                            SET sealed = ?, sealed_hmac = ?, last_seen = ?,
                                sightings = ?
                            WHERE id = ?
                            """,
                            (blob, new_mac, now_iso,
                             rec["sightings"], row_id),
                        )
                        self._audit("unknown_faces", row_id, "UPDATE")
                        self._conn.commit()
                        return row_id

            # No match → new intruder record
            payload = {
                "encoding": enc.tolist(),
                "sightings": 1,
                "meta": metadata or {},
            }
            blob, mac = self._seal_record(payload)
            cur = self._conn.execute(
                """
                INSERT INTO unknown_faces
                (sealed, sealed_hmac, created_at, last_seen, sightings)
                VALUES (?, ?, ?, ?, 1)
                """,
                (blob, mac, now_iso, now_iso),
            )
            row_id = int(cur.lastrowid)
            self._audit("unknown_faces", row_id, "INSERT")
            self._conn.commit()
        return row_id

    def load_unknown(self) -> tuple[list, list]:
        """Decrypt all live unknown templates.

        Returns (encodings, [ {"id", "sightings", "last_seen", "meta"} ]).
        """
        encodings, infos = [], []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, sealed, sealed_hmac, last_seen, sightings
                FROM unknown_faces WHERE deleted = 0
                """
            ).fetchall()
        for row_id, sealed, mac, last_seen, sightings in rows:
            try:
                rec = self._unseal_record(sealed, mac)
            except (VaultIntegrityError, VaultTamperError):
                continue
            encodings.append(np.array(rec["encoding"], dtype=np.float64))
            infos.append({
                "id": row_id,
                "sightings": sightings,
                "last_seen": last_seen,
                "meta": rec.get("meta", {}),
            })
        return encodings, infos

    def enforce_unknown_retention(
        self, days: int = VAULT_UNKNOWN_RETENTION_DAYS
    ) -> int:
        """Tombstone unknown records older than *days*; audit each."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        removed = 0
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, last_seen FROM unknown_faces
                WHERE deleted = 0 AND last_seen < ?
                """,
                (cutoff,),
            ).fetchall()
            for row_id, _ in rows:
                self._conn.execute(
                    "UPDATE unknown_faces SET deleted = 1 WHERE id = ?",
                    (row_id,),
                )
                self._audit("unknown_faces", row_id, "DELETE")
                removed += 1
            if removed:
                self._conn.commit()
        return removed

    def cap_unknown_per_person(
        self, max_records: int = VAULT_MAX_UNKNOWN_PER_PERSON
    ) -> int:
        """Keep at most *max_records* per distinct intruder cluster.

        Simple guard against unbounded growth: after dedup, the newest
        records win; older ones are tombstoned.
        """
        # Dedup already collapses repeats into one row with a sighting
        # counter, so this cap is a safety net for drifted encodings.
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM unknown_faces WHERE deleted = 0"
            ).fetchone()[0]
        if total <= max_records:
            return 0
        removed = 0
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id FROM unknown_faces WHERE deleted = 0
                ORDER BY last_seen DESC
                """
            ).fetchall()
            for row_id, in rows[max_records:]:
                self._conn.execute(
                    "UPDATE unknown_faces SET deleted = 1 WHERE id = ?",
                    (row_id,),
                )
                self._audit("unknown_faces", row_id, "DELETE")
                removed += 1
            if removed:
                self._conn.commit()
        return removed

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def verify_integrity(self) -> list[str]:
        """Full tamper scan: audit chain + every row's HMAC seal.

        Returns a list of human-readable problems; empty = vault is
        clean. Checks:
          1. Every audit entry's HMAC recomputes correctly.
          2. Chain links match (entry N's prev_hmac == entry N-1's hmac).
          3. Every live row's sealed_hmac matches its ciphertext.
        """
        problems: list[str] = []

        # 1 + 2: the chain itself
        with self._lock:
            entries = self._conn.execute(
                """
                SELECT seq, table_name, row_id, action, prev_hmac, entry_hmac
                FROM audit_chain ORDER BY seq
                """
            ).fetchall()
        prev_hmac = b"\x00" * 32
        for seq, table, row_id, action, stored_prev, entry_hmac in entries:
            expected = self.cipher.hmac(
                f"{table}|{row_id}|{action}|".encode("utf-8") + stored_prev
            )
            if entry_hmac != expected:
                problems.append(
                    f"audit_chain seq={seq}: entry HMAC mismatch "
                    f"(possible forged audit entry)"
                )
            if stored_prev != prev_hmac:
                problems.append(
                    f"audit_chain seq={seq}: chain link broken — "
                    f"an entry was deleted or reordered"
                )
            prev_hmac = entry_hmac

        # 3: row seals
        for table in ("family_faces", "unknown_faces"):
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT id, sealed, sealed_hmac FROM {table} "
                    f"WHERE deleted = 0"
                ).fetchall()
            for row_id, sealed, mac in rows:
                if not self.cipher.hmac_verify(sealed, mac):
                    problems.append(
                        f"{table} id={row_id}: row seal mismatch "
                        f"(record edited outside the vault)"
                    )

        # 4: hard-deletion detection — every audited INSERT must still
        # point at a row (live OR tombstoned). A row that vanished
        # outright was deleted by someone bypassing the vault API.
        for table in ("family_faces", "unknown_faces"):
            with self._lock:
                audited = self._conn.execute(
                    "SELECT DISTINCT row_id FROM audit_chain "
                    "WHERE table_name = ? AND action = 'INSERT'",
                    (table,),
                ).fetchall()
                existing = {
                    row[0] for row in self._conn.execute(
                        f"SELECT id FROM {table}"
                    ).fetchall()
                }
            for (row_id,) in audited:
                if row_id not in existing:
                    problems.append(
                        f"{table} id={row_id}: row was deleted "
                        f"outside the vault (missing but audited)"
                    )

        return problems

    # ------------------------------------------------------------------
    # Migration from the legacy photo folders
    # ------------------------------------------------------------------

    def migrate_family_photos(self, family_dir: str) -> int:
        """One-time import: encrypt every legacy family photo.

        Reads ``family/<Name>/*.jpg`` exactly like the old
        load_family_database(), seals each encoding into the vault,
        then renames the folder to ``family.imported/`` so photos are
        never double-imported. Returns the number imported.
        """
        from cctv.faces import load_family_database  # local: avoid cycle

        encodings, names = load_family_database(family_dir)
        imported = 0
        for enc, name in zip(encodings, names):
            self.add_family_face(name, enc)
            imported += 1
        if imported:
            backup = family_dir.rstrip("/\\") + ".imported"
            os.rename(family_dir, backup)
        return imported

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


# ----------------------------------------------------------------------
# Convenience: module-level default vault (lazy, thread-safe)
# ----------------------------------------------------------------------

_default_vault: FaceVault | None = None
_default_lock = threading.Lock()


def get_vault(cipher: VaultCipher | None = None) -> FaceVault:
    """Return the shared FaceVault, creating it on first use."""
    global _default_vault
    with _default_lock:
        if _default_vault is None:
            _default_vault = FaceVault(cipher=cipher)
        return _default_vault
