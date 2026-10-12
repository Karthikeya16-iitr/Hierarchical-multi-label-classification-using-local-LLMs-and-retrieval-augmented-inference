"""Export prediction results to Excel while strictly preserving original inputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

logger = logging.getLogger("mavenmagnet.exporter")


class ResultExporter:
    """Exports model classifications to production Excel workbooks."""

    def __init__(self, output_dir: Path | str) -> None:
        """Initialize exporter with destination directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_predictions(
        self,
        posts: List[str],
        predictions: List[Set[Tuple[str, str]]],
        filename: str = "Process_Classified.xlsx",
    ) -> Path:
        """Export review predictions into an Excel workbook.

        Expands multi-label associations into rows while preserving post text.
        Never overwrites or modifies the original input workbooks.

        Args:
            posts: Original review texts.
            predictions: Predicted sets of (Level 1, Level 2) tuples.
            filename: Output filename.

        Returns:
            Pathlib Path to the generated workbook.
        """
        output_path = self.output_dir / filename
        rows = []

        for post, pred_pairs in zip(posts, predictions):
            if not pred_pairs:
                # Post with no predicted association
                rows.append({"Post": post, "Level 1": "", "Level 2": ""})
            else:
                for l1, l2 in sorted(pred_pairs):
                    rows.append({"Post": post, "Level 1": l1, "Level 2": l2})

        df_out = pd.DataFrame(rows)
        logger.info("Writing %d classification rows to %s", len(df_out), output_path)
        df_out.to_excel(output_path, index=False)
        return output_path
