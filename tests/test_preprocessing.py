import unittest

import pandas as pd

from src.data.preprocessing import (
    temporal_split,
    validate_splits,
    prepare_interactions,
    build_mappings,
    get_evaluation_users,
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.ratings = pd.DataFrame({
            "userId": [1, 1, 2, 2, 1, 3, 2, 3, 1, 2],
            "movieId": [10, 11, 10, 12, 13, 11, 14, 15, 16, 17],
            "rating": [5, 4, 3, 5, 4, 5, 5, 2, 5, 4],
            "timestamp": list(range(100, 110))
        })

        self.movies = pd.DataFrame({
            "movieId": list(range(10, 18)),
            "title": [f"Movie {i}" for i in range(10, 18)],
            "genres": ["Action"] * 8
        })

        users, movies = build_mappings(
            self.ratings,
            self.movies
        )

        self.interactions = prepare_interactions(
            self.ratings,
            users,
            movies
        )

    def test_positive_threshold(self):
        positives = self.interactions[
            self.interactions["is_positive"] == 1
        ]

        self.assertTrue(
            (positives["rating"] >= 4.0).all()
        )

    def test_mapping_is_unique(self):
        self.assertFalse(
            self.interactions["interaction_id"]
            .duplicated()
            .any()
        )

    def test_temporal_order(self):
        train, val, test = temporal_split(
            self.interactions
        )

        self.assertLess(
            train["timestamp"].max(),
            val["timestamp"].min()
        )

        self.assertLess(
            val["timestamp"].max(),
            test["timestamp"].min()
        )

    def test_no_overlap(self):
        train, val, test = temporal_split(
            self.interactions
        )

        validate_splits(
            train,
            val,
            test,
            self.interactions
        )

    def test_all_interactions_preserved(self):
        train, val, test = temporal_split(
            self.interactions
        )

        self.assertEqual(
            len(train) + len(val) + len(test),
            len(self.interactions)
        )

    def test_timestamp_ties_not_split(self):
        df = self.interactions.copy()
        df.loc[7, "timestamp"] = df.loc[6, "timestamp"]

        train, val, test = temporal_split(df)

        timestamp_sets = [
            set(split["timestamp"])
            for split in [train, val, test]
        ]

        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(
                    timestamp_sets[i] & timestamp_sets[j]
                )

    def test_evaluation_user_flags(self):
        train, val, test = temporal_split(
            self.interactions
        )

        users = get_evaluation_users(
            train, val, test
        )

        self.assertIn("eligible_val", users.columns)
        self.assertIn("eligible_test", users.columns)

        self.assertEqual(
            len(users),
            self.ratings["userId"].nunique()
        )


if __name__ == "__main__":
    unittest.main()
