# arxiv_scraper.py
import arxiv
import os 
from typing import List
from utils import logger, time_it

class ArxivScraper:
    def __init__(self):
        pass # No configuration needed for arxiv library in this initial version

    @time_it
    def search_papers(self, search_query: str, max_results: int = 10) -> List[arxiv.Result]:
        """
        Searches arXiv for papers based on the search query.

        Args:
            search_query: The query string to search arXiv.
            max_results: Maximum number of results to return.

        Returns:
            A list of arxiv.Result objects representing the found papers.
        """
        try:
            logger.info(f"Searching arXiv for: '{search_query}'...")
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance # Sort by relevance for initial search
            )
            results = list(search.results()) # Need to convert generator to list to be time_it friendly
            logger.info(f"Found {len(results)} papers on arXiv for query: '{search_query}'.")
            return results
        except Exception as e:
            logger.error(f"Error during arXiv search: {e}")
            return []

    @time_it
    def download_paper_pdf(self, result: arxiv.Result, download_path: str = "arxiv_papers") -> str:
        """
        Downloads the PDF of an arXiv paper.

        Args:
            result: An arxiv.Result object representing the paper.
            download_path: Path to the directory where PDFs will be downloaded.

        Returns:
            The file path to the downloaded PDF, or None if download fails.
        """
        try:
            os.makedirs(download_path, exist_ok=True) # Create directory if it doesn't exist
            filename = f"{result.entry_id.split('/')[-1]}.pdf" # Just filename
            # filepath = os.path.join(download_path, filename) # Not needed anymore with dirpath

            logger.info(f"Downloading paper: '{result.title}' from arXiv...")
            result.download_pdf(dirpath=download_path, filename=filename) # Use dirpath and filename
            filepath = os.path.join(download_path, filename) # Construct filepath now after download
            logger.info(f"Paper '{result.title}' downloaded to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error downloading paper '{result.title}': {e}")
            return None

    def get_paper_metadata(self, result: arxiv.Result) -> dict:
        """
        Extracts relevant metadata from an arxiv.Result object.

        Args:
            result: An arxiv.Result object.

        Returns:
            A dictionary containing paper metadata.
        """
        return {
            "source_document_title": result.title,
            "source_document_type": "arXiv Paper",
            "source_url": result.pdf_url,
            "publication_date": str(result.published), # or result.updated for updated date
            "authors": [str(author) for author in result.authors], # Convert Author objects to strings
            "citation_string": f"{', '.join([str(author) for author in result.authors])} ({result.published.year}). {result.title}. arXiv preprint arXiv:{result.entry_id}", # Basic arXiv citation
            "arxiv_entry_id": result.entry_id, # Store arXiv ID
            "arxiv_primary_category": str(result.primary_category),
            "arxiv_categories": [str(cat) for cat in result.categories],
            "summary": result.summary
            # Add more metadata fields as needed from arxiv.Result object
        }


# Example usage (for testing):
if __name__ == "__main__":
    scraper = ArxivScraper()
    search_term = "large language models"
    papers = scraper.search_papers(search_term, max_results=2)

    if papers:
        for paper in papers:
            print("\n--- Paper ---")
            print("Title:", paper.title)
            print("Authors:", paper.authors)
            print("Published:", paper.published)
            print("Summary:", paper.summary[:200] + "...") # Print first 200 chars of summary
            pdf_path = scraper.download_paper_pdf(paper)
            if pdf_path:
                print("PDF downloaded to:", pdf_path)
            metadata = scraper.get_paper_metadata(paper)
            print("Metadata:", metadata)
    else:
        print(f"No papers found for '{search_term}'.")