import unittest
import numpy as np
from rent_vs_buy import RentVsBuy, annual_to_monthly, monthly_to_annual

LENGTH = 3


class TestRentVsBuy(unittest.TestCase):
    def setUp(self) -> None:
        self.answer = RentVsBuy().calculate().value[0]

    def test_rate_conversions(self):
        test_rate = 0.05
        self.assertTrue(
            np.isclose(monthly_to_annual(annual_to_monthly(test_rate)), test_rate)
        )

    def test_vectorized(self):
        kwargs = {
            k: (np.array([v.default] * LENGTH) if v.type == "ArrayLike" else v.default)
            for k, v in RentVsBuy.RentVsBuyDefaults.__dataclass_fields__.items()
        }

        np.testing.assert_almost_equal(
            RentVsBuy().calculate(**kwargs).value,  # type: ignore
            np.array([self.answer] * LENGTH),
        )

    def test_broadcast(self):
        np.testing.assert_almost_equal(
            RentVsBuy()
            .calculate(
                home_price=np.array([RentVsBuy.RentVsBuyDefaults.home_price] * LENGTH),
            )
            .value,
            np.array([self.answer] * LENGTH),
        )

    def test_ny_times_default(self):
        rent_vs_buy = RentVsBuy().calculate()
        np.testing.assert_almost_equal(910.37, rent_vs_buy.pmt[0][0], decimal=1)
        np.testing.assert_almost_equal(50_000, rent_vs_buy.down_fee[0][0], decimal=1)
        np.testing.assert_almost_equal(
            3362.56, rent_vs_buy.property_taxes[:12].sum(), decimal=1
        )
        np.testing.assert_almost_equal(
            10_000, rent_vs_buy.buying_closing_costs[0][0], decimal=1
        )
        np.testing.assert_almost_equal(
            19571.59, rent_vs_buy.sellers_fee[-1][0], decimal=1
        )
        np.testing.assert_almost_equal(
            2494.75, rent_vs_buy.maintenance[:12].sum(), decimal=1
        )
        np.testing.assert_almost_equal(
            1150.41, rent_vs_buy.insurance[:12].sum(), decimal=1
        )
        np.testing.assert_almost_equal(
            140.02, rent_vs_buy.renters_insurance[:12].sum(), decimal=1
        )


if __name__ == "__main__":
    unittest.main()
