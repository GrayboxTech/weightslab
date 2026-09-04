"""weightslab.export -- annotation export to relabeling-tool formats.

Public API
----------
SUPPORTED_FORMATS : tuple -- ("cvat", "label_studio", "v7")
export_annotations : func -- collect + encode, returns (bytes, filename, mime_type, image_count)
save_export : func -- export_annotations() + write to disk, returns the path written
collect_image_annotations : func -- IR extraction only (no format encoding)
"""

from weightslab.export.collect import collect_image_annotations # noqa: F401
from weightslab.export.exporter import ( # noqa: F401
    SUPPORTED_FORMATS,
    export_annotations,
    save_export,
)
