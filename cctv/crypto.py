"""Authenticated encryption for biometric data (the vault's crypto core).

Face encodings are biometric identifiers: stealing the database must not
leak whose faces the system knows. Every vault row is therefore stored
as a single AES-256-GCM *sealed box*:

    salt(16) + nonce(12) + ciphertext + tag(16)

AES-GCM gives confidentiality **and** integrity in one pass — flipping a
single bit of ciphertext makes decryption fail loudly, so an attacker
cannot rewrite an encoding into someone else's face.

Key management (two sources, chosen by VAULT_KEY_SOURCE in config.py)
---------------------------------------------------------------------
``keyfile``  A random 32-byte key generated once and stored in
             ``logs/.vault.key`` with 0600 permissions. Zero prompts,
             protects against stolen disks/backups/copies.

``passphrase`` The key is derived from a password via PBKDF2-HMAC-SHA256
             (600k iterations) and NEVER touches the disk. Strongest
             option — even a full disk image cannot decrypt the vault.

The module exposes one high-level object, :class:`VaultCipher`, plus the
HMAC helpers used for the tamper-evidence chain in cctv/vault.py.
"""

import hashlib
import hmac
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config import (
    VAULT_KEY_FILE,
    VAULT_PBKDF2_ITERATIONS,
    VAULT_KEY_SOURCE,
)

_SALT_LEN = 16
_NONCE_LEN = 12
_KEY_LEN = 32  # AES-256
_PBKDF2_DIGEST = "sha256"


class VaultKeyError(RuntimeError):
    """Raised when the vault key is missing, unreadable, or corrupt."""


def _derive_key_from_passphrase(passphrase: str, salt: bytes) -> bytes:
    """Derive a 32-byte AES key from a passphrase (PBKDF2-HMAC-SHA256)."""
    return hashlib.pbkdf2_hmac(
        _PBKDF2_DIGEST,
        passphrase.encode("utf-8"),
        salt,
        VAULT_PBKDF2_ITERATIONS,
        dklen=_KEY_LEN,
    )


def load_or_create_keyfile(path: str = VAULT_KEY_FILE) -> bytes:
    """Return the 32-byte vault key, creating the key file on first use.

    The file is created with 0600 permissions (owner-only) and its
    contents are verified to be exactly 32 bytes before use.
    """
    if os.path.exists(path):
        with open(path, "rb") as fh:
            key = fh.read()
        if len(key) != _KEY_LEN:
            raise VaultKeyError(
                f"Vault key file {path!r} is corrupt "
                f"(expected {_KEY_LEN} bytes, got {len(key)})."
            )
        return key

    key = os.urandom(_KEY_LEN)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Write with 0600: only the owner may read the key.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def derive_key_from_passphrase(passphrase: str) -> bytes:
    """Derive the AES key from a passphrase (no disk storage)."""
    if not passphrase:
        raise VaultKeyError("Vault passphrase cannot be empty.")
    return _derive_key_from_passphrase(passphrase, b"smart-cctv-vault-v1")


class VaultCipher:
    """Encrypts / decrypts vault rows with AES-256-GCM.

    One instance wraps one key. ``seal`` produces a self-contained blob
    (salt + nonce + ciphertext + tag); ``unseal`` verifies the GCM tag
    and raises :class:`VaultTamperError` on any modification.
    """

    def __init__(self, key: bytes):
        if len(key) != _KEY_LEN:
            raise VaultKeyError(
                f"AES key must be exactly {_KEY_LEN} bytes."
            )
        self._aes = AESGCM(key)
        self._key = key

    @classmethod
    def from_keyfile(cls, path: str = VAULT_KEY_FILE) -> "VaultCipher":
        return cls(load_or_create_keyfile(path))

    @classmethod
    def from_passphrase(cls, passphrase: str) -> "VaultCipher":
        return cls(derive_key_from_passphrase(passphrase))

    @classmethod
    def from_config(cls, passphrase: str | None = None) -> "VaultCipher":
        """Build the cipher from VAULT_KEY_SOURCE.

        ``passphrase`` is used when the source is ``"passphrase"``; the
        caller (main.py / register.py) collects it interactively.
        """
        if VAULT_KEY_SOURCE == "passphrase":
            if not passphrase:
                raise VaultKeyError(
                    "VAULT_KEY_SOURCE is 'passphrase' but no passphrase "
                    "was provided."
                )
            return cls.from_passphrase(passphrase)
        return cls.from_keyfile()

    # -- sealing -----------------------------------------------------

    def seal(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        """Encrypt *plaintext*; returns salt|nonce|ciphertext+tag.

        A fresh random salt + nonce per call means the same encoding
        encrypted twice yields different bytes (no equality leaks).
        """
        salt = os.urandom(_SALT_LEN)
        nonce = os.urandom(_NONCE_LEN)
        # AESGCM.encrypt appends the 16-byte auth tag to the ciphertext.
        blob = salt + nonce + self._aes.encrypt(nonce, plaintext, aad)
        return blob

    def unseal(self, blob: bytes, aad: bytes = b"") -> bytes:
        """Decrypt and verify; raises VaultTamperError if altered."""
        if len(blob) < _SALT_LEN + _NONCE_LEN + 16:
            raise VaultTamperError("Sealed blob is too short.")
        salt = blob[:_SALT_LEN]
        nonce = blob[_SALT_LEN:_SALT_LEN + _NONCE_LEN]
        body = blob[_SALT_LEN + _NONCE_LEN:]
        try:
            return self._aes.decrypt(nonce, body, aad)
        except Exception as exc:
            raise VaultTamperError(
                "Vault record failed AES-GCM authentication — "
                "it was modified or the key is wrong."
            ) from exc

    # -- HMAC chain (tamper evidence) ---------------------------------

    def hmac(self, message: bytes) -> bytes:
        """HMAC-SHA256 over *message* using the vault key."""
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def hmac_verify(self, message: bytes, mac: bytes) -> bool:
        """Constant-time check that *mac* matches HMAC(*message*)."""
        return hmac.compare_digest(self.hmac(message), mac)


class VaultTamperError(RuntimeError):
    """Raised when a sealed record fails authentication."""


# Backwards-friendly alias: tamper errors are key/cipher failures too.
VaultKeyError = VaultKeyError
