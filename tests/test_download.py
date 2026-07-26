import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from streamrank.data.download import safe_extract_tar


class DownloadTests(unittest.TestCase):
    def test_safe_extract_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo("../escape.txt")
                payload = b"not allowed"
                info.size = len(payload)
                handle.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError, "path traversal"):
                safe_extract_tar(archive, root / "extract")


if __name__ == "__main__":
    unittest.main()
