"""Builds a minimal, valid single-page PDF at test time so PDF-extractor
tests don't depend on a checked-in binary fixture or a PDF-writing library.
"""


def build_minimal_pdf(text: str) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >>"
        b" /MediaBox [0 0 200 200] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = ("BT /F1 18 Tf 20 100 Td (%s) Tj ET" % text).encode("latin-1")
    objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf += ("%d 0 obj\n" % i).encode() + obj + b"\nendobj\n"

    xref_offset = len(buf)
    buf += ("xref\n0 %d\n" % (len(objects) + 1)).encode()
    buf += b"0000000000 65535 f \n"
    for offset in offsets:
        buf += ("%010d 00000 n \n" % offset).encode()

    buf += b"trailer\n"
    buf += ("<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)).encode()
    buf += b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF"
    return bytes(buf)
