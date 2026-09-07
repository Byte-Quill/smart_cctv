"""Tests for cctv/crypto.py — AES-256-GCM sealing and key management.

These tests run without a camera: they exercise the pure crypto layer
with random keys and synthetic payloads.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from cctv.crypto import (
    VaultCipher,
    VaultKeyError,
    VaultTamperError,
    derive_key_from_passphrase,
    load_or_create_keyfile,
)


def make_cipher() -> VaultCipher:
    return VaultCipher(os.urandom(32))


class TestSealUnseal(unittest.TestCase):

    def test_roundtrip(self):
        c = make_cipher()
        blob = c.seal(b"family face data")
        self.assertEqual(c.unseal(blob), b"family face data")

    def test_blob_is_not_plaintext(self):
        c = make_cipher()
        payload = b"Alice's face encoding"
        blob = c.seal(payload)
        self.assertNotIn(payload, blob)

    def test_same_plaintext_different_blobs(self):
        # Fresh salt+nonce per call → no equality leaks between rows.
        c = make_cipher()
        b1 = c.seal(b"same")
        b2 = c.seal(b"same")
        self.assertNotEqual(b1, b2)

    def test_tampered_blob_raises(self):
        c = make_cipher()
        blob = bytearray(c.seal(b"secret"))
        blob[-1] ^= 0x01  # flip one bit of the GCM tag
        with self.assertRaises(VaultTamperError):
            c.unseal(bytes(blob))

    def test_truncated_blob_raises(self):
        c = make_cipher()
        with self.assertRaises(VaultTamperError):
            c.unseal(b"short")

    def test_wrong_key_raises(self):
        c1 = make_cipher()
        c2 = make_cipher()
        blob = c1.seal(b"secret")
        with self.assertRaises(VaultTamperError):
            c2.unseal(blob)

    def test_aad_mismatch_raises(self):
        c = make_cipher()
        blob = c.seal(b"secret", aad=b"context-A")
        with self.assertRaises(VaultTamperError):
            c.unseal(blob, aad=b"context-B")


class TestKeyManagement(unittest.TestCase):

    def test_keyfile_created_with_owner_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", ".vault.key")
            key = load_or_create_keyfile(path)
            self.assertEqual(len(key), 32)
            # File exists, is exactly 32 bytes, and owner-only readable.
            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.path.getsize(path), 32)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            # Second load returns the SAME key (stable across restarts).
            self.assertEqual(load_or_create_keyfile(path), key)

    def test_corrupt_keyfile_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".vault.key")
            with open(path, "wb") as fh:
                fh.write(b"too short")
            with self.assertRaises(VaultKeyError):
                load_or_create_keyfile(path)

    def test_passphrase_derivation_is_deterministic_and_hard(self):
        k1 = derive_key_from_passphrase("correct horse")
        k2 = derive_key_from_passphrase("correct horse")
        self.assertEqual(k1, k2)  # same passphrase → same key
        self.assertNotEqual(
            derive_key_from_passphrase("wrong battery"), k1
        )

    def test_empty_passphrase_rejected(self):
        with self.assertRaises(VaultKeyError):
            derive_key_from_passphrase("")

    def test_bad_key_length_rejected(self):
        with self.assertRaises(VaultKeyError):
            VaultCipher(b"short")


class TestHMAC(unittest.TestCase):

    def test_hmac_roundtrip(self):
        c = make_cipher()
        mac = c.hmac(b"audit entry")
        self.assertTrue(c.hmac_verify(b"audit entry", mac))
        self.assertFalse(c.hmac_verify(b"audit entry!", mac))

    def test_hmacs_differ_per_cipher(self):
        # Two vaults (different keys) produce different MACs — a hacker
        # cannot forge audit entries without the key.
        mac1 = make_cipher().hmac(b"x")
        mac2 = make_cipher().hmac(b"x")
        self.assertNotEqual(mac1, mac2)


if __name__ == "__main__":
    unittest.main()
