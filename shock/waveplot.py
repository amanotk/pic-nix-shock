import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from . import base
except ImportError:
    import base


class SixPanelWavePlot:
    def __init__(self, options):
        self.options = options
        self.plot_dict = None

    def save(self, filename):
        if self.plot_dict is not None and "fig" in self.plot_dict:
            self.plot_dict["fig"].savefig(filename)
        else:
            print("No plot to save")

    def plot(self, X, Y, time, Az, vars, labels, xlim=None):
        if self.plot_dict is None:
            self.plot_dict = self.plot_new(X, Y, Az, vars, labels, xlim=xlim)
        else:
            self.plot_dict = self.plot_update(X, Y, Az, vars, labels, xlim=xlim)

        plt.suptitle(r"$\Omega_{{ci}} t$ = {:5.2f}".format(time))

    def _get_vlim(self):
        quantity = self.options.get("quantity", "wave")
        if quantity == "field":
            E_lim = self.options.get("E_lim", [-0.2, 0.2])
            B_lim = self.options.get("B_lim", [-0.2, 0.2])
            return [
                E_lim,
                E_lim,
                E_lim,
                B_lim,
                B_lim,
                B_lim,
            ]

        B_wave_lim = self.options.get("B_wave_lim", [-0.25, 0.25])
        B_env_lim = self.options.get("B_env_lim", [0.0, 0.5])
        B_abs_lim = self.options.get("B_abs_lim", [0.5, 5.0])
        S_para_lim = self.options.get("S_para_lim", [-0.5, 0.5])
        wave_layout = self.options.get("wave_layout", "standard")
        if wave_layout == "diagnostics_left":
            return [
                B_abs_lim,
                B_env_lim,
                S_para_lim,
                B_wave_lim,
                B_wave_lim,
                B_wave_lim,
            ]
        return [
            B_wave_lim,
            B_wave_lim,
            B_wave_lim,
            B_abs_lim,
            B_env_lim,
            S_para_lim,
        ]

    def plot_new(self, X, Y, Az, vars, labels, xlim=None):
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()
        if xlim is not None:
            xmin, xmax = xlim

        fig = plt.figure(figsize=(10, 8), dpi=120)
        fig.subplots_adjust(
            top=0.92,
            bottom=0.05,
            left=0.08,
            right=0.90,
            hspace=0.20,
            wspace=0.22,
        )
        gridspec = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[50, 1, 50])
        axs = [0] * 6
        axs[0] = fig.add_subplot(gridspec[0, 0])
        axs[1] = fig.add_subplot(gridspec[1, 0])
        axs[2] = fig.add_subplot(gridspec[2, 0])
        axs[3] = fig.add_subplot(gridspec[0, 2])
        axs[4] = fig.add_subplot(gridspec[1, 2])
        axs[5] = fig.add_subplot(gridspec[2, 2])

        cxs = [0] * 6
        img = [0] * 6
        cnt = [0] * 6
        clb = [0] * 6

        vlim = self._get_vlim()
        common_args = {
            "extent": [xmin, xmax, ymin, ymax],
            "origin": "lower",
        }
        args = [
            {**common_args, "vmin": vlim[0][0], "vmax": vlim[0][1], "cmap": "bwr"},
            {**common_args, "vmin": vlim[1][0], "vmax": vlim[1][1], "cmap": "bwr"},
            {**common_args, "vmin": vlim[2][0], "vmax": vlim[2][1], "cmap": "bwr"},
            {**common_args, "vmin": vlim[3][0], "vmax": vlim[3][1], "cmap": "viridis"},
            {**common_args, "vmin": vlim[4][0], "vmax": vlim[4][1], "cmap": "viridis"},
            {**common_args, "vmin": vlim[5][0], "vmax": vlim[5][1], "cmap": "bwr"},
        ]
        quantity = self.options.get("quantity", "wave")
        if quantity == "field":
            for a in args:
                a["cmap"] = "bwr"
        elif quantity == "wave":
            wave_layout = self.options.get("wave_layout", "standard")
            if wave_layout == "diagnostics_left":
                args[0]["cmap"] = "viridis"
                args[1]["cmap"] = "viridis"
                args[2]["cmap"] = "bwr"
                args[3]["cmap"] = "bwr"
                args[4]["cmap"] = "bwr"
                args[5]["cmap"] = "bwr"

        for i in range(6):
            plt.sca(axs[i])
            img[i] = plt.imshow(vars[i], **args[i])
            cnt[i] = plt.contour(X, Y, Az, levels=25, colors="k", linewidths=0.5)
            axs[i].set_xlim(xmin, xmax)
            axs[i].set_ylim(ymin, ymax)
            axs[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
            axs[i].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
            axs[i].set_aspect("equal")
            cxs[i] = plt.axes(base.get_colorbar_position_next(axs[i], 0.025))
            clb[i] = plt.colorbar(img[i], cax=cxs[i])
            axs[i].set_title(labels[i])

        [axs[i].set_ylabel(r"$y / c/\omega_{pe}$") for i in (0, 1, 2)]
        [axs[i].set_xlabel(r"$x / c/\omega_{pe}$") for i in (2, 5)]

        return {"fig": fig, "axs": axs, "img": img, "cnt": cnt, "cxs": cxs, "clb": clb}

    def plot_update(self, X, Y, Az, vars, labels, xlim=None):
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()
        if xlim is not None:
            xmin, xmax = xlim
        vlim = self._get_vlim()

        fig = self.plot_dict["fig"]
        axs = self.plot_dict["axs"]
        img = self.plot_dict["img"]
        cnt = self.plot_dict["cnt"]
        cxs = self.plot_dict["cxs"]
        clb = self.plot_dict["clb"]

        for i in range(6):
            cnt[i].remove()
        for i in range(6):
            plt.sca(axs[i])
            img[i].set_array(vars[i])
            img[i].set_extent([xmin, xmax, ymin, ymax])
            cnt[i] = plt.contour(X, Y, Az, levels=25, colors="k", linewidths=0.5)
            axs[i].set_xlim(xmin, xmax)
            axs[i].set_ylim(ymin, ymax)
            axs[i].set_title(labels[i])
            img[i].set_clim(vlim[i])
            clb[i].update_normal(img[i])

        return {"fig": fig, "axs": axs, "img": img, "cnt": cnt, "cxs": cxs, "clb": clb}
