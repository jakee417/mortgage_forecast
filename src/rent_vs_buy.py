from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import numpy_financial as npf

from typing import Union
from dataclasses import dataclass

PERIODS: int = 12
SINGLE_STANDARD_DEDUCTION: int = 12_950
JOINT_STANDARD_DEDUCTION: int = 25_900
TAX_CREDIT_LIMIT: int = 750_000


def annual_to_monthly(rate: ArrayLike, periods: int = 12) -> ArrayLike:
    return np.add(1.0, rate) ** (1 / periods) - 1.0


def monthly_to_annual(rate: ArrayLike, periods: int = 12) -> ArrayLike:
    return np.add(1.0, rate) ** (periods) - 1.0


class RentVsBuy:

    @property
    def value(self) -> np.ndarray:
        return self.buy_vs_rent[-1]

    @dataclass(frozen=True)
    class RentVsBuyDefaults:
        home_price: ArrayLike = 250_000
        years: int = 9
        mortgage_rate: ArrayLike = 0.0367
        downpayment: ArrayLike = 0.20
        length_of_mortgage: int = 30
        home_price_growth_rate: ArrayLike = 0.03
        rent_growth_rate: ArrayLike = 0.025
        investment_return_rate: ArrayLike = 0.04
        inflation_rate: ArrayLike = 0.02
        filing_jointly: bool = True
        property_tax_rate: ArrayLike = 0.0135
        marginal_tax_rate: ArrayLike = 0.2
        costs_of_buying_home: ArrayLike = 0.04
        costs_of_selling_home: ArrayLike = 0.06
        maintenance_rate: ArrayLike = 0.01
        home_owners_insurance_rate: ArrayLike = 0.0046
        monthly_utilities: ArrayLike = 100
        monthly_common_fees: ArrayLike = 0
        monthly_rent: ArrayLike = 884
        security_deposit: ArrayLike = 1
        brokers_fee: ArrayLike = 0.00
        renters_insurance_rate: ArrayLike = 0.0132

    def calculate(
        self,
        home_price: ArrayLike = RentVsBuyDefaults.home_price,
        years: int = RentVsBuyDefaults.years,
        mortgage_rate: ArrayLike = RentVsBuyDefaults.mortgage_rate,
        downpayment: ArrayLike = RentVsBuyDefaults.downpayment,
        length_of_mortgage: int = RentVsBuyDefaults.length_of_mortgage,
        home_price_growth_rate: ArrayLike = RentVsBuyDefaults.home_price_growth_rate,
        rent_growth_rate: ArrayLike = RentVsBuyDefaults.rent_growth_rate,
        investment_return_rate: ArrayLike = RentVsBuyDefaults.investment_return_rate,
        inflation_rate: ArrayLike = RentVsBuyDefaults.inflation_rate,
        filing_jointly: bool = RentVsBuyDefaults.filing_jointly,
        property_tax_rate: ArrayLike = RentVsBuyDefaults.property_tax_rate,
        marginal_tax_rate: ArrayLike = RentVsBuyDefaults.marginal_tax_rate,
        costs_of_buying_home: ArrayLike = RentVsBuyDefaults.costs_of_buying_home,
        costs_of_selling_home: ArrayLike = RentVsBuyDefaults.costs_of_selling_home,
        maintenance_rate: ArrayLike = RentVsBuyDefaults.maintenance_rate,
        home_owners_insurance_rate: ArrayLike = RentVsBuyDefaults.home_owners_insurance_rate,
        monthly_utilities: ArrayLike = RentVsBuyDefaults.monthly_utilities,
        monthly_common_fees: ArrayLike = RentVsBuyDefaults.monthly_common_fees,
        monthly_rent: ArrayLike = RentVsBuyDefaults.monthly_rent,
        security_deposit: ArrayLike = RentVsBuyDefaults.security_deposit,
        brokers_fee: ArrayLike = RentVsBuyDefaults.brokers_fee,
        renters_insurance_rate: ArrayLike = RentVsBuyDefaults.renters_insurance_rate,
    ) -> RentVsBuy:
        # TODO: Add PMI if downpayment < 0.2
        self = RentVsBuy()
        loan = np.array(home_price * np.subtract(1, downpayment)).reshape(1, -1)
        down = np.subtract(home_price, loan)

        ###########################################################################
        # Convert to monthly rates.
        ###########################################################################
        mortgage_rate = np.array(annual_to_monthly(mortgage_rate)).reshape(1, -1)
        home_price_growth_rate = annual_to_monthly(home_price_growth_rate)
        rent_growth_rate = annual_to_monthly(rent_growth_rate)
        investment_return_rate = annual_to_monthly(investment_return_rate)
        inflation_rate = annual_to_monthly(inflation_rate)
        property_tax_rate = annual_to_monthly(property_tax_rate)
        marginal_tax_rate = annual_to_monthly(marginal_tax_rate)
        maintenance_rate = annual_to_monthly(maintenance_rate)
        home_owners_insurance_rate = annual_to_monthly(home_owners_insurance_rate)

        ###########################################################################
        # Period Counters.
        ###########################################################################
        self.per = np.mgrid[1 : years * PERIODS + 1].reshape(-1, 1)
        self.annual_per = np.repeat(range(years), 12).reshape(-1, 1)
        self.mortgage_per = np.mgrid[1 : length_of_mortgage * PERIODS + 1].reshape(
            -1, 1
        )
        self.per_inv = (years * PERIODS - self.per).reshape(-1, 1)

        ###########################################################################
        # Home Value.
        ###########################################################################
        self.home_value = npf.fv(
            rate=home_price_growth_rate,
            nper=self.per,
            pmt=0,
            pv=np.multiply(home_price, -1),
        )

        ###########################################################################
        # Home Liabilities.
        ###########################################################################
        # down_fee will have the downpayment in first row, for each column.
        # Pad the rest of the downpayment with zeros.
        self.down_fee = np.append(
            down, np.zeros((years * PERIODS - 1, down.shape[1])), axis=0
        )
        self.monthly_fees = npf.fv(
            rate=monthly_to_annual(inflation_rate),
            nper=self.annual_per,
            pmt=0,
            pv=np.multiply(monthly_common_fees, -1),
        )
        self.home_monthly_utilities = npf.fv(
            rate=monthly_to_annual(inflation_rate),
            nper=self.annual_per,
            pmt=0,
            pv=np.multiply(monthly_utilities, -1),
        )
        self.first_month_home_value = np.repeat(
            self.home_value[::PERIODS], PERIODS, axis=0
        )
        self.property_taxes = np.multiply(
            self.first_month_home_value, property_tax_rate
        )
        self.maintenance = np.multiply(self.first_month_home_value, maintenance_rate)
        self.insurance = np.multiply(
            self.first_month_home_value, home_owners_insurance_rate
        )

        # home_price * costs_of_buying_home in first row, for each column.
        # Pad the rest of the buying_closing_costs with zeros.
        self.buying_closing_costs = (
            np.array(home_price).reshape(1, -1) * costs_of_buying_home
        )
        self.buying_closing_costs = np.append(
            self.buying_closing_costs,
            np.zeros((years * PERIODS - 1, self.buying_closing_costs.shape[1])),
            axis=0,
        )

        # Since the mortgage timeline can be different than the home holding period,
        # we have to fill in the lesser of the two with the payments.
        slice = np.s_[: min(years, length_of_mortgage) * PERIODS]

        # Our result for ppmt, ipmt, and pmt can either be 1-d or 2-d depending on inputs.
        payment_shape = (years * PERIODS, max(loan.shape[-1], mortgage_rate.shape[-1]))

        # Create the principal, interest and total payments.
        self.ppmt = np.zeros(payment_shape)
        self.ppmt[slice] = npf.ppmt(
            rate=mortgage_rate,
            per=self.mortgage_per,
            nper=length_of_mortgage * PERIODS,
            pv=np.multiply(loan, -1),
        )[slice]
        self.ipmt = np.zeros(payment_shape)
        self.ipmt[slice] = npf.ipmt(
            rate=mortgage_rate,
            per=self.mortgage_per,
            nper=length_of_mortgage * PERIODS,
            pv=np.multiply(loan, -1),
        )[slice]
        cu_ppmt = np.cumsum(self.ppmt, axis=0)
        self.pmt = self.ipmt + self.ppmt
        self.total_home_liability = (
            self.pmt
            + self.maintenance
            + self.insurance
            + self.property_taxes
            + self.monthly_fees
            + self.buying_closing_costs
            + self.down_fee
            + self.home_monthly_utilities
        )

        ###########################################################################
        # Home Assets.
        ###########################################################################
        annual_ipmt = self.ipmt.reshape(-1, PERIODS, self.ipmt.shape[1]).sum(1)
        tax_credit_limit = annual_ipmt.cumsum(axis=0) <= TAX_CREDIT_LIMIT
        annual_ipmt *= tax_credit_limit
        standard_deduction = npf.fv(
            rate=monthly_to_annual(inflation_rate),
            nper=np.arange(years).reshape(-1, 1),
            pmt=0,
            pv=-(
                JOINT_STANDARD_DEDUCTION
                if filing_jointly
                else SINGLE_STANDARD_DEDUCTION
            ),
        )
        self.mortgage_interest_contribution = marginal_tax_rate * np.maximum(
            annual_ipmt - standard_deduction,
            0,
        )
        self.total_home_assets = np.zeros((years * PERIODS, self.ipmt.shape[1]))
        self.total_home_assets[
            PERIODS - 1 :: PERIODS, :
        ] += self.mortgage_interest_contribution

        ###########################################################################
        # Rent Liabilities.
        ###########################################################################
        self.rent = npf.fv(
            rate=monthly_to_annual(rent_growth_rate),
            nper=self.annual_per,
            pmt=0,
            pv=np.multiply(monthly_rent, -1),
        )
        self.renters_insurance = np.multiply(self.rent, renters_insurance_rate)
        self.brokers_fee_cost = np.multiply(self.rent, brokers_fee)
        self.security_deposit = np.array(security_deposit).reshape(1, -1)
        self.monthly_rent = np.array(monthly_rent).reshape(1, -1)
        self.security_deposit_cost = np.zeros(
            (
                years * PERIODS,
                max(self.security_deposit.shape[1], self.monthly_rent.shape[1]),
            )
        )
        self.security_deposit_cost[0, :] = np.multiply(
            self.security_deposit, self.monthly_rent
        )
        self.total_rent_liability = (
            self.rent
            + self.renters_insurance
            + self.security_deposit
            + self.brokers_fee_cost
        )

        ###########################################################################
        # Rent Assets.
        ###########################################################################
        self.total_rent_assets = np.zeros((years * PERIODS, 1))

        ###########################################################################
        # Opportunity Cost.
        ###########################################################################
        self.home_opportunity_cost = np.maximum(
            (self.total_home_liability - self.total_home_assets)
            - (self.total_rent_liability - self.total_rent_assets),
            0,
        )
        self.home_opportunity_cost_fv = npf.fv(
            rate=investment_return_rate,
            nper=self.per_inv,
            pmt=0,
            pv=-self.home_opportunity_cost,
        )
        self.home_cumulative_opportunity = np.cumsum(
            self.home_opportunity_cost_fv, axis=0
        )
        self.rental_opportunity_cost = np.maximum(
            (self.total_rent_liability - self.total_rent_assets)
            - (self.total_home_liability - self.total_home_assets),
            0,
        )
        self.rental_opportunity_cost_fv = npf.fv(
            rate=investment_return_rate,
            nper=self.per_inv,
            pmt=0,
            pv=-self.rental_opportunity_cost,
        )
        self.rental_cumulative_opportunity = np.cumsum(
            self.rental_opportunity_cost_fv, axis=0
        )
        self.selling_closing_costs = np.multiply(self.home_value, costs_of_selling_home)
        self.equity = (
            self.home_value - np.subtract(loan, cu_ppmt) - self.selling_closing_costs
        )
        self.buy_vs_rent = (
            self.equity
            + self.rental_cumulative_opportunity
            - self.home_cumulative_opportunity
        )
        return self


def rent_vs_buy_breakeven_objective_closure(scalar: str, **kwargs):

    def rent_vs_buy_breakeven_objective(x: Union[int, float]):
        kwargs.update({scalar: x})
        # Only scalar output is supported for minimization.
        return abs(float(RentVsBuy().calculate(**kwargs).value) - 1.0)

    return rent_vs_buy_breakeven_objective
