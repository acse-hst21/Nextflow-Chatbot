# backend/app/document_fetcher.py
import os
import requests
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class GitHubDocumentFetcher:
    """Fetches markdown documents from Nextflow GitHub repository."""

    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize with optional GitHub token for higher rate limits.

        Args:
            github_token: Optional GitHub personal access token
        """
        self.github_token = github_token
        self.base_url = "https://api.github.com"
        self.repo_owner = "nextflow-io"
        self.repo_name = "nextflow"
        self.docs_path = "docs"
        self.session = requests.Session()

        if self.github_token:
            self.session.headers.update({
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            })

    def _make_request(self, url: str) -> Dict[Any, Any]:
        """Make authenticated request to GitHub API."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from GitHub API: {e}")
            raise

    def _get_directory_contents(self, path: str = "") -> List[Dict[str, Any]]:
        """Get contents of a directory in the repository."""
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/contents"
        if path:
            url += f"/{path}"

        return self._make_request(url)

    def _get_file_content(self, file_info: Dict[str, Any]) -> str:
        """Get the content of a file from GitHub."""
        if file_info.get("encoding") == "base64":
            content = base64.b64decode(file_info["content"]).decode("utf-8")
            return content
        else:
            # Fallback to download_url if not base64 encoded
            response = requests.get(file_info["download_url"])
            response.raise_for_status()
            return response.text

    def _extract_markdown_files_recursive(self, path: str = "") -> List[Dict[str, Any]]:
        """Recursively extract all markdown files from the docs directory."""
        markdown_files = []

        try:
            contents = self._get_directory_contents(f"{self.docs_path}/{path}" if path else self.docs_path)

            for item in contents:
                if item["type"] == "file" and item["name"].endswith(".md"):
                    # Get file content
                    file_content = self._get_file_content(item)

                    # Build GitHub URL for the file
                    github_url = f"https://github.com/{self.repo_owner}/{self.repo_name}/blob/master/{item['path']}"

                    markdown_files.append({
                        "path": item["path"],
                        "name": item["name"],
                        "content": file_content,
                        "github_url": github_url,
                        "size": item["size"],
                        "sha": item["sha"]
                    })

                elif item["type"] == "dir":
                    # Recursively process subdirectories
                    subdir_path = f"{path}/{item['name']}" if path else item["name"]
                    markdown_files.extend(self._extract_markdown_files_recursive(subdir_path))

        except Exception as e:
            print(f"Error processing directory {path}: {e}")

        return markdown_files

    def fetch_all_markdown_documents(self) -> List[Dict[str, Any]]:
        """
        Fetch all markdown documents from the Nextflow docs directory.

        Returns:
            List of document dictionaries with content and metadata
        """
        print(f"[{datetime.utcnow().isoformat()}] Starting to fetch Nextflow documentation...")

        try:
            markdown_files = self._extract_markdown_files_recursive()

            documents = []
            for file_info in markdown_files:
                # Extract title from filename or first heading
                title = self._extract_title(file_info["content"], file_info["name"])

                doc = {
                    "id": f"nextflow_docs_{file_info['sha'][:8]}",
                    "title": title,
                    "text": file_info["content"],
                    "url": file_info["github_url"],
                    "source_path": file_info["path"],
                    "file_name": file_info["name"],
                    "size": file_info["size"],
                    "fetched_at": datetime.utcnow().isoformat()
                }
                documents.append(doc)

            print(f"[{datetime.utcnow().isoformat()}] Successfully fetched {len(documents)} markdown documents")
            return documents

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Error fetching documents: {e}")
            raise

    def _extract_title(self, content: str, filename: str) -> str:
        """Extract title from document content or use filename as fallback."""
        lines = content.split('\n')

        # Look for first heading
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('## '):
                return line[3:].strip()

        # Fallback to filename without extension
        return filename.replace('.md', '').replace('_', ' ').replace('-', ' ').title()


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment variables."""
    return os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_ACCESS_TOKEN")


# Convenience function for easy importing
def fetch_nextflow_docs() -> List[Dict[str, Any]]:
    """Fetch all Nextflow documentation with optional GitHub token."""
    token = get_github_token()
    fetcher = GitHubDocumentFetcher(token)
    return fetcher.fetch_all_markdown_documents()