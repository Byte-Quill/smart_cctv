"""Tests for cctv/vault.py — encrypted, tamper-evident face storage.

Runs without a camera: uses synthetic 128-d encodings and a temporary
vault database per test.
"""

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cctv.crypto import VaultCipher
from cctv.vault import FaceVault, VaultIntegrityError


def make_vault(tmpdir: str) -> FaceVault:
    return FaceVault(
        path=os.path.join(tmpdir, "vault.db"),
        cipher=VaultCipher(os.urandom(32)),
    )


def make_encoding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=128)


class TestFamilyFaces(unittest.TestCase):

    def test_add_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            enc = make_encoding(1)
            vault.add_family_face("Alice", enc)
            encs, names = vault.load_family()
            self.assertEqual(names, ["Alice"])
            self.assertEqual(len(encs), 1)
            np.testing.assert_allclose(encs[0], enc)
            vault.close()

    def test_multiple_people_and_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.add_family_face("Alice", make_encoding(1))
            vault.add_family_face("Alice", make_encoding(2))
            vault.add_family_face("Bob", make_encoding(3))
            encs, names = vault.load_family()
            self.assertEqual(sorted(names), ["Alice", "Alice", "Bob"])
            self.assertEqual(len(encs), 3)
            vault.close()

    def test_stored_rows_are_encrypted(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.add_family_face("Alice", make_encoding(1))
            # Raw DB inspection: neither the name nor the encoding may
            # appear in plaintext inside the vault file.
            with open(vault.path, "rb") as fh:
                raw = fh.read()
            self.assertNotIn(b"Alice", raw)
            vault.close()

    def test_delete_family_person_tombstones(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.add_family_face("Alice", make_encoding(1))
            vault.add_family_face("Bob", make_encoding(2))
            removed = vault.delete_family_person("Alice")
            self.assertEqual(removed, 1)
            encs, names = vault.load_family()
            self.assertEqual(names, ["Bob"])
            # Tombstoned row still exists on disk (soft delete) but is
            # excluded from live queries.
            with sqlite3.connect(vault.path) as conn:
                n = conn.execute(
                    "SELECT COUNT(*) FROM family_faces WHERE deleted = 1"
                ).fetchone()[0]
            self.assertEqual(n, 1)
            vault.close()


class TestUnknownFaces(unittest.TestCase):

    def test_record_and_load_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.record_unknown(make_encoding(10), {"mode": "NIGHT"})
            encs, infos = vault.load_unknown()
            self.assertEqual(len(encs), 1)
            self.assertEqual(infos[0]["sightings"], 1)
            self.assertEqual(infos[0]["meta"]["mode"], "NIGHT")
            vault.close()

    def test_repeat_intruder_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            enc = make_encoding(10)
            vault.record_unknown(enc)
            vault.record_unknown(enc)  # same stranger again
            encs, infos = vault.load_unknown()
            self.assertEqual(len(encs), 1)
            self.assertEqual(infos[0]["sightings"], 2)
            vault.close()

    def test_different_strangers_are_separate_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.record_unknown(make_encoding(10))
            vault.record_unknown(make_encoding(11))  # different person
            encs, _ = vault.load_unknown()
            self.assertEqual(len(encs), 2)
            vault.close()

    def test_retention_tombstones_old_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.record_unknown(make_encoding(10))
            # Force last_seen into the past.
            with sqlite3.connect(vault.path) as conn:
                conn.execute(
                    "UPDATE unknown_faces SET last_seen = '2000-01-01T00:00:00'"
                )
            removed = vault.enforce_unknown_retention(days=30)
            self.assertEqual(removed, 1)
            encs, _ = vault.load_unknown()
            self.assertEqual(len(encs), 0)
            vault.close()

    def test_cap_unknown_per_person(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            for i in range(10):
                vault.record_unknown(make_encoding(100 + i))
            removed = vault.cap_unknown_per_person(max_records=3)
            self.assertGreater(removed, 0)
            encs, _ = vault.load_unknown()
            self.assertEqual(len(encs), 3)
            vault.close()


class TestTamperEvidence(unittest.TestCase):

    def test_clean_vault_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.add_family_face("Alice", make_encoding(1))
            vault.record_unknown(make_encoding(2))
            self.assertEqual(vault.verify_integrity(), [])
            vault.close()

    def test_edited_row_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.add_family_face("Alice", make_encoding(1))
            # Attacker flips a byte of the ciphertext directly in the
            # database (bypassing the vault API).
            with sqlite3.connect(vault.path) as conn:
                blob = conn.execute(
                    "SELECT sealed FROM family_faces WHERE id = 1"
                ).fetchone()[0]
                tampered = bytes(blob[:-1]) + bytes([blob[-1] ^ 0x01])
                conn.execute(
                    "UPDATE family_faces SET sealed = ? WHERE id = 1",
                    (tampered,),
                )
            problems = vault.verify_integrity()
            self.assertTrue(any("family_faces id=1" in p for p in problems))
            vault.close()

    def test_deleted_row_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.add_family_face("Alice", make_encoding(1))
            vault.add_family_face("Bob", make_encoding(2))
            # Attacker deletes a row outright (not via tombstone).
            with sqlite3.connect(vault.path) as conn:
                conn.execute("DELETE FROM family_faces WHERE id = 1")
            problems = vault.verify_integrity()
            self.assertTrue(
                any("deleted outside the vault" in p for p in problems)
            )
            vault.close()

    def test_forged_audit_entry_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.add_family_face("Alice", make_encoding(1))
            # Attacker injects a fake audit entry with a bogus HMAC.
            with sqlite3.connect(vault.path) as conn:
                conn.execute(
                    """
                    INSERT INTO audit_chain
                    (table_name, row_id, action, prev_hmac, entry_hmac)
                    VALUES ('family_faces', 99, 'DELETE', ?, ?)
                    """,
                    (b"A" * 32, b"B" * 32),
                )
            problems = vault.verify_integrity()
            self.assertTrue(any("HMAC mismatch" in p for p in problems))
            vault.close()

    def test_tombstone_via_api_is_not_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = make_vault(tmp)
            vault.add_family_face("Alice", make_encoding(1))
            vault.delete_family_person("Alice")
            # Legitimate admin deletion keeps the chain intact.
            self.assertEqual(vault.verify_integrity(), [])
            vault.close()


if __name__ == "__main__":
    unittest.main()
