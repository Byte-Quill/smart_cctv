"""Encrypted face vault administration tool.

The vault (logs/vault.db) holds the family's biometric templates and the
unknown-person records — all AES-256-GCM encrypted. This CLI is the
*only* supported way to inspect and manage them:

    python vault_admin.py list                 # family members + counts
    python vault_admin.py unknown               # intruder records
    python vault_admin.py remove <Name>         # tombstone a family member
    python vault_admin.py verify                # full tamper-evidence scan
    python vault_admin.py stats                  # vault summary

Why an admin tool?
------------------
The vault is fail-closed: main.py refuses to run if its HMAC chain shows
tampering. When you legitimately need to remove a family member (someone
moved out) or audit who the camera caught, you do it here — every action
is itself written to the audit chain, so even admin deletions leave
tamper-evident traces.
"""

import sys

from config import VAULT_KEY_SOURCE

from cctv.vault import FaceVault


def _open_vault() -> FaceVault:
    """Open the vault, prompting for a passphrase when configured."""
    if VAULT_KEY_SOURCE == "passphrase":
        import getpass

        from cctv.crypto import VaultCipher

        passphrase = getpass.getpass("Vault passphrase: ")
        return FaceVault(cipher=VaultCipher.from_passphrase(passphrase))
    return FaceVault()


def cmd_list(vault: FaceVault) -> int:
    encodings, names = vault.load_family()
    if not names:
        print("Vault holds no family templates.")
        return 0

    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1

    print(f"Family members in vault ({len(counts)}):")
    for name in sorted(counts):
        print(f"  {name:<20} {counts[name]} template(s)")
    print(f"Total: {len(names)} encrypted template(s)")
    return 0


def cmd_unknown(vault: FaceVault) -> int:
    encodings, infos = vault.load_unknown()
    if not infos:
        print("No unknown-person records in the vault.")
        return 0

    print(f"Unknown-person records ({len(infos)}):")
    for info in infos:
        meta = info.get("meta", {})
        mode = meta.get("mode", "?")
        print(
            f"  id={info['id']:<4} sightings={info['sightings']:<4} "
            f"last_seen={info['last_seen']}  mode={mode}"
        )
    return 0


def cmd_remove(vault: FaceVault, name: str) -> int:
    removed = vault.delete_family_person(name)
    if removed:
        print(
            f"Tombstoned {removed} template(s) of '{name}'. "
            f"The deletion is recorded in the audit chain."
        )
    else:
        print(f"No live templates found for '{name}'.")
    return 0


def cmd_verify(vault: FaceVault) -> int:
    problems = vault.verify_integrity()
    if not problems:
        print("Vault integrity: OK — no tampering detected.")
        return 0
    print(f"Vault integrity: FAILED ({len(problems)} problem(s)):")
    for problem in problems:
        print(f"  • {problem}")
    return 1


def cmd_stats(vault: FaceVault) -> int:
    encodings, names = vault.load_family()
    uenc, uinfos = vault.load_unknown()
    print("Vault summary")
    print(f"  family templates : {len(encodings)}")
    people = sorted(set(names))
    print(f"  family members    : {len(people)}"
          + (f" ({', '.join(people)})" if people else ""))
    print(f"  unknown records   : {len(uinfos)}")
    total_sightings = sum(i["sightings"] for i in uinfos)
    print(f"  unknown sightings : {total_sightings}")
    return 0


def main() -> int:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    command = args[0]

    try:
        vault = _open_vault()
    except Exception as err:
        print(f"ERROR: could not open the vault: {err}")
        return 2

    try:
        if command == "list":
            return cmd_list(vault)
        if command == "unknown":
            return cmd_unknown(vault)
        if command == "remove":
            if len(args) < 2:
                print("Usage: python vault_admin.py remove <Name>")
                return 2
            return cmd_remove(vault, args[1])
        if command == "verify":
            return cmd_verify(vault)
        if command == "stats":
            return cmd_stats(vault)
        print(f"Unknown command: {command!r}\n")
        print(__doc__)
        return 2
    finally:
        vault.close()


if __name__ == "__main__":
    sys.exit(main())
