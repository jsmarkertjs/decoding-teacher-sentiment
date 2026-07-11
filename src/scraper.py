"""
RateMyProfessors.com GraphQL Scraper

Scrapes professor profiles and text reviews from RateMyProfessors.com
using their internal GraphQL API with curl_cffi for TLS fingerprint impersonation.

Based on the approach from github.com/ppannuta/ratemyprofessor-api.
"""

from curl_cffi import requests
import pandas as pd
import base64
import time


class Professor:
    """Represents a professor with their metadata from RMP."""

    def __init__(self, ratemyprof_id: str, first_name: str, last_name: str,
                 num_of_ratings: int, overall_rating):
        self.ratemyprof_id = ratemyprof_id
        self.name = f"{first_name} {last_name}"
        self.num_of_ratings = num_of_ratings
        self.overall_rating = 0.0 if self.num_of_ratings < 1 else float(overall_rating)


class RateMyProfApi:
    """Main API client for scraping RateMyProfessors.com."""

    def __init__(self, school_id: str):
        self.UniversityId = school_id
        school_string = f"School-{self.UniversityId}"
        self.graphql_school_id = base64.b64encode(
            school_string.encode('utf-8')
        ).decode('utf-8')

        self.headers = {
            "Authorization": "Basic dGVzdDp0ZXN0",
            "Content-Type": "application/json"
        }

        self.professors = self.scrape_professors()
        self.professorlist = list(self.professors.values())

    def scrape_professors(self):
        """Scrape the master list of all professors at the school."""
        professors = dict()

        query = """
        query TeacherSearchPaginationQuery($count: Int!, $cursor: String, $query: TeacherSearchQuery!) {
          newSearch {
            teachers(query: $query, first: $count, after: $cursor) {
              didFallback
              resultCount
              pageInfo { hasNextPage endCursor }
              edges {
                node { id legacyId firstName lastName numRatings avgRating }
              }
            }
          }
        }
        """

        variables = {
            "count": 20,
            "cursor": "",
            "query": {
                "text": "",
                "schoolID": self.graphql_school_id,
                "fallback": True
            }
        }

        has_next_page = True
        pages_scraped = 0

        print("Gathering master list of professors from the server...")

        while has_next_page:
            response = requests.post(
                "https://www.ratemyprofessors.com/graphql",
                headers=self.headers,
                json={"query": query, "variables": variables},
                impersonate="chrome"
            )

            try:
                data = response.json()
            except Exception:
                print("Failed to parse JSON. Cloudflare might be blocking the request.")
                break

            if "errors" in data:
                print("GraphQL Error:", data["errors"])
                break

            teachers_data = data.get("data", {}).get("newSearch", {}).get("teachers", {})
            if not teachers_data:
                break

            total_expected = teachers_data.get("resultCount", "Unknown")

            for edge in teachers_data["edges"]:
                node = edge["node"]
                if node["numRatings"] > 0:
                    prof = Professor(
                        node["legacyId"],
                        node["firstName"],
                        node["lastName"],
                        node["numRatings"],
                        node["avgRating"]
                    )
                    professors[prof.ratemyprof_id] = prof

            pages_scraped += 1
            if pages_scraped % 10 == 0:
                print(
                    f"--> Found {len(professors)} professors with ratings "
                    f"so far (Out of ~{total_expected} total)..."
                )

            has_next_page = teachers_data["pageInfo"]["hasNextPage"]
            variables["cursor"] = teachers_data["pageInfo"]["endCursor"]

            time.sleep(0.1)

        print(
            f"Finished. Successfully compiled a list of "
            f"{len(professors)} professors with ratings."
        )
        return professors

    def create_reviews_list(self, tid):
        """Fetch all text reviews for a single professor by their legacy ID."""
        reviews_list = []
        teacher_string = f"Teacher-{tid}"
        graphql_tid = base64.b64encode(
            teacher_string.encode('utf-8')
        ).decode('utf-8')

        query = """
        query TeacherRatingsPaginationQuery($count: Int!, $cursor: String, $id: ID!) {
          node(id: $id) {
            ... on Teacher {
              ratings(first: $count, after: $cursor) {
                pageInfo { hasNextPage endCursor }
                edges {
                  node {
                    class
                    comment
                    difficultyRating
                    clarityRating
                    date
                    grade
                    ratingTags
                    wouldTakeAgain
                  }
                }
              }
            }
          }
        }
        """
        variables = {"count": 100, "cursor": "", "id": graphql_tid}

        has_next_page = True
        while has_next_page:
            response = requests.post(
                "https://www.ratemyprofessors.com/graphql",
                headers=self.headers,
                json={"query": query, "variables": variables},
                impersonate="chrome"
            )
            try:
                ratings_data = response.json()["data"]["node"]["ratings"]
            except (TypeError, KeyError):
                break

            for edge in ratings_data["edges"]:
                reviews_list.append(edge["node"])

            has_next_page = ratings_data["pageInfo"]["hasNextPage"]
            variables["cursor"] = ratings_data["pageInfo"]["endCursor"]

        return reviews_list

    def get_all_university_reviews(self) -> pd.DataFrame:
        """Scrape all reviews for every professor and return as a DataFrame."""
        all_reviews = []
        total_profs = len(self.professorlist)

        print(
            f"\nFetching text reviews for {total_profs} professors. "
            f"This will take several minutes..."
        )

        for index, prof in enumerate(self.professorlist):
            if index % 50 == 0 and index > 0:
                print(
                    f"--> Processed reviews for {index}/{total_profs} professors..."
                )

            prof_reviews = self.create_reviews_list(prof.ratemyprof_id)

            for review in prof_reviews:
                wta_raw = review.get("wouldTakeAgain")
                if wta_raw is True or wta_raw == 1:
                    wta_clean = "Yes"
                elif wta_raw is False or wta_raw == 0:
                    wta_clean = "No"
                else:
                    wta_clean = "N/A"

                tags_raw = review.get("ratingTags")
                if isinstance(tags_raw, str):
                    tags_clean = tags_raw.replace("--", ", ")
                elif isinstance(tags_raw, list):
                    tags_clean = ", ".join(tags_raw)
                else:
                    tags_clean = ""

                review_data = {
                    "professor_id": prof.ratemyprof_id,
                    "professor_name": prof.name,
                    "course": review.get("class"),
                    "review_text": review.get("comment"),
                    "grade": review.get("grade"),
                    "difficulty": review.get("difficultyRating"),
                    "quality": review.get("clarityRating"),
                    "would_take_again": wta_clean,
                    "tags": tags_clean,
                    "date": review.get("date")
                }
                all_reviews.append(review_data)

            time.sleep(0.1)

        return pd.DataFrame(all_reviews)


if __name__ == '__main__':
    # American University's school ID is "32"
    AU_SCHOOL_ID = "32"

    print("Initializing scraper for American University...")
    au_api = RateMyProfApi(AU_SCHOOL_ID)

    df_all_reviews = au_api.get_all_university_reviews()

    print(f"\nScraping complete!")
    print(f"Total reviews extracted: {len(df_all_reviews)}")
    print(df_all_reviews.head(10))
