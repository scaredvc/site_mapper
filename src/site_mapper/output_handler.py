import json
import csv
from pathlib import Path
from typing import Dict, Any, List

class OutputHandler:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, data: Dict[str, Any], filename: str = "crawl_results.json"):
        """Save results as JSON"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_csv(self, link_graph: Dict[str, List[Dict]], filename: str = "crawl_results.csv"):
        """Save results as CSV"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['source_url', 'target_url', 'link_text', 'is_external'])
            # Write data
            for source_url, links in link_graph.items():
                for link in links:
                    writer.writerow([
                        source_url,
                        link['absolute_url'],
                        link['text'],
                        link['is_external']
                    ])