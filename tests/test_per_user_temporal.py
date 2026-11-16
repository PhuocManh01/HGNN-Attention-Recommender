
import unittest

import pandas as pd

from src.data.preprocessing import (
    per_user_temporal_split,
    validate_splits,
    get_evaluation_users,
)


class TestPerUserTemporal(unittest.TestCase):

    def make_data(self):
        rows = []
        interaction_id = 0

        for user in range(3):
            for i in range(20):
                rows.append({
                    "interaction_id": interaction_id,
                    "user_idx": user,
                    "movie_idx": user * 100 + i,
                    "timestamp": 1000 + i,
                    "is_positive": 1,
                })
                interaction_id += 1

        return pd.DataFrame(rows)

    def test_no_interaction_loss(self):
        data = self.make_data()

        train, val, test = (
            per_user_temporal_split(data)
        )

        self.assertEqual(
            len(train) + len(val) + len(test),
            len(data),
        )

        validate_splits(
            train,
            val,
            test,
            data,
            split_method="per_user_temporal",
        )

    def test_every_user_has_history(self):
        data = self.make_data()

        train, val, test = (
            per_user_temporal_split(data)
        )

        self.assertEqual(
            set(train["user_idx"]),
            {0, 1, 2},
        )
        self.assertEqual(
            set(val["user_idx"]),
            {0, 1, 2},
        )
        self.assertEqual(
            set(test["user_idx"]),
            {0, 1, 2},
        )

    def test_expected_ratios(self):
        data = self.make_data()

        train, val, test = (
            per_user_temporal_split(data)
        )

        self.assertEqual(len(train), 48)
        self.assertEqual(len(val), 6)
        self.assertEqual(len(test), 6)

    def test_eligibility(self):
        data = self.make_data()

        train, val, test = (
            per_user_temporal_split(data)
        )

        users = get_evaluation_users(
            train, val, test
        )

        self.assertEqual(
            int(users["eligible_val"].sum()),
            3,
        )
        self.assertEqual(
            int(users["eligible_test"].sum()),
            3,
        )

    def test_identical_timestamps_stay_together(self):
        data = self.make_data()

        # Introduce tied timestamps within user 0.
        mask = (
            (data["user_idx"] == 0)
            & data["timestamp"].isin([1015, 1016])
        )
        data.loc[mask, "timestamp"] = 1015

        train, val, test = (
            per_user_temporal_split(data)
        )

        validate_splits(
            train, val, test, data,
            split_method="per_user_temporal",
        )

        partitions = [train, val, test]
        locations = [
            i for i, part in enumerate(partitions)
            if (
                (part["user_idx"] == 0)
                & (part["timestamp"] == 1015)
            ).any()
        ]

        self.assertEqual(len(locations), 1)

    def test_short_history_remains_in_train(self):
        data = self.make_data()

        short_user = pd.DataFrame([
            {
                "interaction_id": 1000,
                "user_idx": 99,
                "movie_idx": 999,
                "timestamp": 1000,
                "is_positive": 1,
            },
            {
                "interaction_id": 1001,
                "user_idx": 99,
                "movie_idx": 998,
                "timestamp": 1001,
                "is_positive": 1,
            },
        ])

        data = pd.concat(
            [data, short_user],
            ignore_index=True,
        )

        train, val, test = (
            per_user_temporal_split(data)
        )

        self.assertIn(99, set(train["user_idx"]))
        self.assertNotIn(99, set(val["user_idx"]))
        self.assertNotIn(99, set(test["user_idx"]))

    def test_reproducibility(self):
        data = self.make_data()

        first = per_user_temporal_split(data)
        second = per_user_temporal_split(data)

        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
