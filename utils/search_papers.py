import time
import logging
from scholarly import scholarly

logger = logging.getLogger(__name__)


def search_papers(query, max_results=5):
    """
    Search Google Scholar for papers relevant to the query.
    Returns a list of dicts: [{title, authors, year, url, abstract_snippet}]

    Note: scholarly can be rate-limited by Google Scholar.
    A short sleep between results reduces the chance of being blocked.
    """
    papers = []
    try:
        search_results = scholarly.search_pubs(query)
        for i, result in enumerate(search_results):
            if i >= max_results:
                break
            bib = result.get("bib", {})
            authors = bib.get("author", ["Unknown"])
            papers.append({
                "title": bib.get("title", "Unknown title"),
                "authors": authors if isinstance(authors, list) else [authors],
                "year": bib.get("pub_year", "Unknown"),
                "url": result.get("pub_url") or result.get("eprint_url", ""),
                "abstract_snippet": bib.get("abstract", "")[:300],
            })
            time.sleep(1)  # polite delay to avoid rate limiting
    except Exception as e:
        logger.warning(f"Paper search failed: {e}")

    return papers


if __name__ == "__main__":
    results = search_papers("RMSD drift molecular dynamics protein simulation", max_results=3)
    for p in results:
        authors_str = ", ".join(p["authors"][:3])
        print(f"\nTitle:    {p['title']}")
        print(f"Authors:  {authors_str}")
        print(f"Year:     {p['year']}")
        print(f"URL:      {p['url']}")
        print(f"Abstract: {p['abstract_snippet'][:150]}")
