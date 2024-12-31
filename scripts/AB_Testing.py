import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind

def perform_ab_testing(df):
    """
    Perform A/B hypothesis testing on the insurance data.
    """
    # Null Hypotheses to test
    null_hypotheses = [
        "There are no risk differences across provinces",
        "There are no risk differences between zip codes",
        "There are no significant margin (profit) difference between zip codes",
        "There are not significant risk difference between Women and Men"
    ]

    # Perform tests and report results
    for hypothesis in null_hypotheses:
        print(f"Testing hypothesis: {hypothesis}")
        
        if "risk differences" in hypothesis:
            if "provinces" in hypothesis:
                # Test risk differences across provinces
                risk_by_province = df.groupby("Province")["TotalClaims"].mean()
                _, p_value, _, _ = chi2_contingency(pd.crosstab(df["Province"], df["TotalClaims"] > 0))
                
            else:
                # Test risk differences between zip codes
                risk_by_zipcode = df.groupby("PostalCode")["TotalClaims"].mean()
                _, p_value, _, _ = chi2_contingency(pd.crosstab(df["PostalCode"], df["TotalClaims"] > 0))
                
            if p_value < 0.05:
                print("Reject the null hypothesis. There are significant risk differences.")
            else:
                print("Fail to reject the null hypothesis. There are no significant risk differences.")
                
        elif "margin (profit) difference" in hypothesis:
            # Test margin (profit) difference between zip codes
            profit_by_zipcode = df.groupby("PostalCode")["TotalPremium"].sum() - df.groupby("PostalCode")["TotalClaims"].sum()
            _, p_value = ttest_ind(profit_by_zipcode.loc[profit_by_zipcode.index % 2 == 0], 
                                  profit_by_zipcode.loc[profit_by_zipcode.index % 2 != 0])
            
            if p_value < 0.05:
                print("Reject the null hypothesis. There are significant margin differences between zip codes.")
            else:
                print("Fail to reject the null hypothesis. There are no significant margin differences between zip codes.")
                
        elif "risk difference between Women and Men" in hypothesis:
            # Test risk difference between genders
            risk_by_gender = df.groupby("Gender")["TotalClaims"].mean()
            _, p_value, _, _ = chi2_contingency(pd.crosstab(df["Gender"], df["TotalClaims"] > 0))
            
            if p_value < 0.05:
                print("Reject the null hypothesis. There are significant risk differences between genders.")
            else:
                print("Fail to reject the null hypothesis. There are no significant risk differences between genders.")
