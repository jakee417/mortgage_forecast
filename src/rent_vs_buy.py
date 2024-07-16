from __future__ import annotations
import dataclasses

import numpy as np
from numpy.typing import ArrayLike
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Tuple, Union
from dataclasses import Field, dataclass
from IPython.display import display, Markdown

PERIODS: int = 12
SINGLE_STANDARD_DEDUCTION: int = 12_950
JOINT_STANDARD_DEDUCTION: int = 25_900
TAX_CREDIT_LIMIT: int = 750_000
PMI_CUTOFF = 0.2


# Helper functions.
def render_plotly_html(fig: go.Figure) -> None:
    """Display a Plotly figure in markdown with HTML."""
    # Show the figure if in a IPython display.
    fig.show()
    # Render the content for html output.
    display(
        Markdown(
            fig.to_html(
                include_plotlyjs="cdn",
            )
        )
    )


def annual_to_monthly(rate: ArrayLike, periods: int = 12) -> ArrayLike:
    return np.add(1.0, rate) ** (1 / periods) - 1.0


def monthly_to_annual(rate: ArrayLike, periods: int = 12) -> ArrayLike:
    return np.add(1.0, rate) ** (periods) - 1.0


@dataclass
class RentVsBuy:
    # Variables
    annual_per: ArrayLike = 0
    per: ArrayLike = 0

    # Home Value
    home_value: ArrayLike = 0
    first_month_home_value: ArrayLike = 0

    # Home Liabilities
    ppmt: ArrayLike = 0
    ipmt: ArrayLike = 0
    pmt: ArrayLike = 0
    maintenance: ArrayLike = 0
    insurance: ArrayLike = 0
    property_taxes: ArrayLike = 0
    monthly_common_fees: ArrayLike = 0
    buying_closing_costs: ArrayLike = 0
    down_fee: ArrayLike = 0
    home_monthly_utilities: ArrayLike = 0
    pmi: ArrayLike = 0
    sellers_fee: ArrayLike = 0
    loan_payoff: ArrayLike = 0
    total_home_liability: ArrayLike = 0

    # Home Assets
    total_home_assets: ArrayLike = 0

    # Rent Liabilities
    rent: ArrayLike = 0
    renters_insurance: ArrayLike = 0
    security_deposit_cost: ArrayLike = 0
    brokers_fee_cost: ArrayLike = 0
    total_rent_liability: ArrayLike = 0

    # Rent Assets
    total_rent_assets: ArrayLike = 0

    # Opportunity Cost
    home_opportunity_cost: ArrayLike = 0
    rental_opportunity_cost: ArrayLike = 0
    home_opportunity_cost_fv: ArrayLike = 0
    rental_opportunity_cost_fv: ArrayLike = 0
    home_opportunity_cost_fv_post_tax: ArrayLike = 0
    rental_opportunity_cost_fv_post_tax: ArrayLike = 0
    home_cumulative_opportunity: ArrayLike = 0
    rental_cumulative_opportunity: ArrayLike = 0
    buy_vs_rent: ArrayLike = 0

    @property
    def fields(self) -> Tuple[Field[Any], ...]:
        return dataclasses.fields(self)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {i.name: getattr(self, i.name).flatten() for i in self.fields}
        )

    @property
    def value(self) -> np.ndarray:
        return self.buy_vs_rent[-1]

    @dataclass(frozen=True)
    class RentVsBuyDefaults:
        home_price: ArrayLike = 250_000
        years: int = 9
        mortgage_rate: ArrayLike = 0.0367
        downpayment: ArrayLike = 0.20
        pmi: ArrayLike = 0.005
        length_of_mortgage: int = 30
        home_price_growth_rate: ArrayLike = 0.03
        rent_growth_rate: ArrayLike = 0.025
        investment_return_rate: ArrayLike = 0.04
        investment_tax_rate: ArrayLike = 0.15
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
        pmi: ArrayLike = RentVsBuyDefaults.pmi,
        length_of_mortgage: int = RentVsBuyDefaults.length_of_mortgage,
        home_price_growth_rate: ArrayLike = RentVsBuyDefaults.home_price_growth_rate,
        rent_growth_rate: ArrayLike = RentVsBuyDefaults.rent_growth_rate,
        investment_return_rate: ArrayLike = RentVsBuyDefaults.investment_return_rate,
        investment_tax_rate: ArrayLike = RentVsBuyDefaults.investment_tax_rate,
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
        """Calculate the opportunity cost of buying a house.

        Notes:
            Arguments can be scalar value or 1-D and broadcastable against each other.

            See `RentVsBuy.RentVsBuyDefaults` for default values.

        Args:
            home_price: the original price of the home.
            years: number of years expecting to keep the home.
            mortgage_rate: mortgage rate to finance the home.
            downpayment: home price as downpayment expressed as a percentage.
            pmi: primary mortgage insurance expressed as a percentage of total loan.
            length_of_mortgage: length of home loan.
            home_price_growth_rate: rate at which the value of the home increases.
            rent_growth_rate: rate at which the cost of rent increases.
            investment_return_rate: rate of making money on a counterfactual investment.
            inflation_rate: rate of inflation.
            filing_jointly: whether you file jointly for taxes.
            property_tax_rate: rate of property taxes.
            marginal_tax_rate: rate of marginal tax
            costs_of_buying_home: costs of buying home expressed as a percentage.
            costs_of_selling_home: costs of selling home expressed as a percentage.
            maintenance_rate: costs of maintaining a home expressed as a percentage.
            home_owners_insurance_rate: rate of homeowners insurance.
            monthly_utilities: cost of monthly utilities as monthly price.
            monthly_common_fees: cost of common fees (i.e. HOA) as a monthly price.
            monthly_rent: cost of rent as a monthly price.
            security_deposit: security deposit expressed as a percentage.
            brokers_fee: brokers fee expressed as a percentage.
            renters_insurance_rate: renters insurance rate expressed as a percentage.

        Returns:
            Rent vs. buy analysis.
        """
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
        # years must be an int for proper indexing.
        years = np.round(years).astype(int).squeeze()
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
        self.monthly_common_fees = npf.fv(
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

        # Calculate the PMI payment
        remaining_home_price_percentage = np.divide((down + cu_ppmt), home_price)
        pmi_mask = remaining_home_price_percentage < PMI_CUTOFF
        self.pmi = np.multiply(pmi_mask, pmi) * loan / PERIODS

        # Sellers fee
        self.sellers_fee = np.zeros_like(self.home_value)
        self.sellers_fee[-1, :] = np.multiply(
            self.home_value[-1, :], np.array(costs_of_selling_home)
        )

        # Need to pay any remaining part of the loan
        self.loan_payoff = np.zeros_like(cu_ppmt)
        self.loan_payoff[-1, :] = loan - cu_ppmt[-1, :]

        self.total_home_liability = (
            self.pmt
            + self.maintenance
            + self.insurance
            + self.property_taxes
            + self.monthly_common_fees
            + self.buying_closing_costs
            + self.down_fee
            + self.home_monthly_utilities
            + self.pmi
            + self.sellers_fee
            + self.loan_payoff
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
        self.total_home_assets = np.zeros((years * PERIODS, self.home_value.shape[1]))
        # Tax credit at the end of the year for paying mortgage
        self.total_home_assets[PERIODS - 1 :: PERIODS, :] = np.add(
            self.total_home_assets[PERIODS - 1 :: PERIODS, :],
            self.mortgage_interest_contribution,
        )
        # Selling the home in the last period (minus sellers cost).
        self.total_home_assets[-1, :] += self.home_value[-1, :]

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
            + self.security_deposit_cost
            + self.brokers_fee_cost
        )

        ###########################################################################
        # Rent Assets.
        ###########################################################################
        self.total_rent_assets = np.zeros((years * PERIODS, 1))
        # We get back the security deposit
        self.total_rent_assets[-1, :] = np.multiply(
            self.security_deposit, self.monthly_rent
        )

        ###########################################################################
        # Opportunity Cost.
        ###########################################################################
        self.home_opportunity_cost = np.maximum(
            (self.total_home_liability - self.total_home_assets)
            - (self.total_rent_liability - self.total_rent_assets),
            0,
        )
        self.rental_opportunity_cost = np.maximum(
            (self.total_rent_liability - self.total_rent_assets)
            - (self.total_home_liability - self.total_home_assets),
            0,
        )

        # Compute the future value of these cash flow but apply a tax rate for the earnings.
        self.home_opportunity_cost_fv = npf.fv(
            rate=investment_return_rate,
            nper=self.per_inv,
            pmt=0,
            pv=-self.home_opportunity_cost,
        )
        self.home_opportunity_cost_fv_post_tax = np.maximum(
            np.multiply(
                self.home_opportunity_cost_fv,
                (1.0 - np.array(investment_tax_rate)),
            ),
            self.home_opportunity_cost,
        )
        self.rental_opportunity_cost_fv = npf.fv(
            rate=investment_return_rate,
            nper=self.per_inv,
            pmt=0,
            pv=-self.rental_opportunity_cost,
        )
        self.rental_opportunity_cost_fv_post_tax = np.maximum(
            np.multiply(
                self.rental_opportunity_cost_fv,
                (1.0 - np.array(investment_tax_rate)),
            ),
            self.rental_opportunity_cost,
        )

        # Compute cumulative values of the cash flow's future value.
        self.rental_cumulative_opportunity = np.cumsum(
            self.rental_opportunity_cost_fv_post_tax, axis=0
        )
        self.home_cumulative_opportunity = np.cumsum(
            self.home_opportunity_cost_fv_post_tax, axis=0
        )

        # Buy vs. Rent is our equity - costs - opportunity cost
        self.buy_vs_rent = (
            self.rental_cumulative_opportunity - self.home_cumulative_opportunity
        )
        return self


def rent_vs_buy_breakeven_objective_closure(scalar: str, **kwargs):

    def _fn(x: Union[int, float]):
        kwargs.update({scalar: x})
        # Only scalar output is supported for minimization.
        return abs(float(RentVsBuy().calculate(**kwargs).value) - 1.0)

    return _fn


def rent_vs_buy_objective_closure(scalar: str, maximize: bool, **kwargs):

    def _fn(x: Union[int, float]):
        kwargs.update({scalar: x})
        # Only scalar output is supported for minimization.
        return RentVsBuy().calculate(**kwargs).value * (-1 if maximize else 1.0)

    return _fn
