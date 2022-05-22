
import pandas_profiling
def get_profile(df):
    with open("output/findings/EDAReport.html","w+") as f:
        pandas_profiling.ProfileReport(df, title="Pandas Profiling Report", explorative=True).to_file("output/findings/EDAReport.html")