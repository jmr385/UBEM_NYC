



# PLOT ONE BETA DISTRIBUTION PLOT
beta_distribution(betas, 9)



# PLOT BETA DISTRIBUTIONS & BUILDING HOURLY LOADS
plot_2x2_hourly_load(ubem, chrystler_building_hourly, start=7*24, end=7*24+168, duration=168)
plot_2x2_hourly_load(ubem, chrystler_building_hourly, start=152*24 + 3*24, end=152*24 + 3*24+168, duration=168)
[beta_distribution(betas, i) for i in range(25)]
np.sum(betas[:,18])



from itertools import product
fig11 = plt.figure(figsize=(15, 15), constrained_layout=False)

# gridspec inside gridspec
outer_grid = fig11.add_gridspec(5, 5, wspace=0.1, hspace=0.1)

for p in range(25):
    beta1 = betas[:, p]
    beta2 = betas[:, p+25]
    beta3 = betas[:, p+50]
    bins = np.linspace(0, 1.1, 11)  #
    weights = np.ones_like(beta1) / float(len(beta1))
    # inner_grid = outer_grid[p].subgridspec(3, 3, wspace=0.0, hspace=0.0)
    ax = fig11.add_subplot(outer_grid[p])
    ax.hist([beta1, beta2, beta3], bins-0.05, weights=[weights, weights, weights], color=['#388F7B', '#594E6A', '#F45539'])
    ax.set_ylim([0, 1.0])

    if p in [0, 1, 2, 3, 10, 11, 12, 14, 17, 18, 22, 23]:
        ax.set_facecolor('#AAB8B1')
    elif p in [9, 19, 20, 21]:
        ax.set_facecolor('#FDE6A9')
    elif p in [4, 5, 6]:
        ax.set_facecolor('#E3E9E8')
    else:
        ax.set_facecolor('#DBCDC0')

    if p >= 20 :
        ax.xaxis.set_ticks_position('bottom')
        # ax.set_xticks(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        # ax.set_xticks([0, 0.33, 0.5, 0.66, 0.75, 1.1])
    else:
        ax.xaxis.set_ticklabels([])
        # ax.set_xticks([])

    if p % 5 == 0:
        ax.yaxis.set_ticks_position('left')
        # ax.set_yticks(bins)
    else:
        ax.set_yticks([])

    fig11.add_subplot(ax)
# plt.show()

all_axes = fig11.get_axes()

# show only the outside spines
for ax in all_axes:
    for sp in ax.spines.values():
        # sp.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # if ax.is_first_row():
    #     ax.spines['top'].set_visible(True)
    # if ax.is_last_row():
    #     ax.spines['bottom'].set_visible(True)
    #     ax.xaxis.set_ticks_position('bottom')
    # if ax.is_first_col():
    #     ax.spines['left'].set_visible(True)
    #     ax.xaxis.set_ticks_position('bottom')
    # if ax.is_last_col():
    #     ax.spines['right'].set_visible(True)


plt.savefig('Figures/Beta_5x5_Histograms.pdf')
plt.show()



