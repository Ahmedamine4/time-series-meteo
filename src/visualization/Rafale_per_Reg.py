import matplotlib.pyplot as plt

def plot_gusts_by_station(df):
    counts = df.groupby("indicatif")["has_gust"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))

    counts.plot(kind="bar", ax=ax)

    ax.set_title("Number of Wind Gusts per Weather Station")
    ax.set_xlabel("Weather Station")
    ax.set_ylabel("Number of Gusts")

    plt.tight_layout()

    return fig