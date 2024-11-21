import sunpy
import sunpy.map
import matplotlib.pyplot as plt
import eispac 
import os 
import numpy as np
import astropy.constants as const
from astropy.visualization import (ImageNormalize, AsinhStretch,
                                   ZScaleInterval)
import astropy.units as u
from astropy.coordinates import SpectralCoord
from matplotlib.ticker import AutoMinorLocator
import matplotlib.cbook as cbook
import matplotlib.lines as mlines
import types
import argparse

'''
This script provides a naive GUI to inspect EIS data or fit files. Can be called from the command line or imported as a module.

Example:

Inspect window 0 line profiles of a data file:
    python eispac_inspector.py -d path_to_data_file -i index_of_window

Inspect window containing wavelength 195.119 Angstroms of a data file:
    python eispac_inspector.py -d path_to_data_file -i 195.119

Inspect component 0 line profile of a fit file (automatically search for the corresponding data file):
    python eispac_inspector.py -f path_to_fit_file -i 0

Inspect component 0 line profile of a fit file and the corresponding data file:
    python eispac_inspector.py -d path_to_data_file -f path_to_fit_file -i 0
'''


class EISInspector:

    '''
    A class to inspect EIS data or fit files.

    Parameters:
    -----------
    filename_data: str
        The path to the data file. If only a data file is provided, the GUI will show the integrated intensity map. 
        If both data and fit files are provided, the GUI will show the fitted intensity, velocity, and width maps.
        Default is None.
    filename_fit: str
        The path to the fit file. If only a fit file is provided, the routine will search for the corresponding data file in the same directory, 
        or the file path restored in the fit file. Otherwise, you must provide the data file.
        Default is None.
    index: int
        The index of the window/wavelength (data file) or component (fitting file) to inspect. When inspecting a data file,
        the index can be the index of the window or the wavelength of the window. When inspecting a fit file, the index is the component index
        of the multi-Gaussian fit.
        Default is 0.

    Attributes:
    -----------
    filename_data: str
        The path to the data file.
    filename_fit: str
        The path to the fit file.
    index: int
        The index of the window/wavelength (data file) or component (fitting file) to inspect.
    data_type: str
        The type of the file, either "data" or "fit".
    data: eispac.core.EISCube
        The EIS data cube.
    fit: eispac.core.EISFitResult
        The EIS fit object.
    intmap: sunpy.map.Map
        The integrated/fitted intensity map.
    velmap: sunpy.map.Map
        The Doppler velocity map.
    widmap: sunpy.map.Map
        The line width map in effective velocity (1/e width).
    xy_ratio: float
        The aspect ratio of the map.
    fig: matplotlib.figure.Figure
        The figure object.
    ax1: matplotlib.axes.Axes
        The intensity map axis.
    ax2: matplotlib.axes.Axes
        The velocity map axis.
    ax3: matplotlib.axes.Axes
        The width map axis.
    ax_line_profile: matplotlib.axes.Axes
        The line profile axis.
    ax_residual: matplotlib.axes.Axes
        The residual axis.
    line_data: matplotlib.container.ErrorbarContainer
        The line profile of the data.
    line_data_fill: matplotlib.collections.PolyCollection
        The filled area of the data line profile.
    line_all_fit: matplotlib.lines.Line2D
        The line profile of the fit.
    line_single_fits: list
        The line profiles of the individual Gaussian components.
    line_residual: matplotlib.container.ErrorbarContainer
        The residual line profile.
    text: matplotlib.text.Text
        The text object showing the intensity, velocity, and width of the selected pixel.
    select_x: int
        The x-coordinate of the selected pixel.
    select_y: int
        The y-coordinate of the selected pixel.
    skycoord: astropy.coordinates.SkyCoord
        The sky coordinate of the selected pixel.
    markers: list
        The marker object of the selected pixel.
    '''
    def __init__(self, filename_data=None, filename_fit=None, index=0):
        if filename_data is None and filename_fit is None:
            raise ValueError("You must provide either a data or fit file.")
        else:
            self.filename_data = filename_data
            self.filename_fit = filename_fit

        self.index = index

        self._load_data()
        self._init_gui()
    

    def _load_data(self):
        '''
        Load the data or fit file. Phrase the data or fit file, e.g., removing instrumental width from the line width.
        '''
        if self.filename_data is not None and self.filename_fit is None:
            self.data_type = "data"
            self.data = eispac.read_cube(self.filename_data, window=self.index)
            self.xy_ratio = self.data.meta["mod_index"]["fovx"]/self.data.meta["mod_index"]["fovy"]
            data_to_mean = self.data.data.copy()
            data_to_mean[data_to_mean < 0] = np.nan
            self.intmap = sunpy.map.Map(np.nanmean(self.data.data, axis=-1), self.data.wcs.celestial)

        elif self.filename_fit is not None:
            self.data_type = "fit"
            self.fit = eispac.read_fit(self.filename_fit)
            self.intmap = self.fit.get_map(component=self.index, measurement="intensity")
            self.velmap = self.fit.get_map(component=self.index, measurement="vel")
            self.widmap = self.fit.get_map(component=self.index, measurement="wid")
            self.xy_ratio = self.intmap.meta["fovx"]/self.intmap.meta["fovy"]

            c = const.c.cgs.value
            amu = const.u.cgs.value
            k_B = const.k_B.cgs.value

            rest_wvl = np.float64(self.intmap.meta["line_id"].split(" ")[-1])

            true_width_fwhm = np.sqrt( (self.widmap.data * np.sqrt(8*np.log(2)))**2 - self.fit.meta["slit_width"][:,np.newaxis]**2)
            v1oe = true_width_fwhm/np.sqrt(4*np.log(2))*c/rest_wvl

            self.widmap = sunpy.map.Map(v1oe/1e5, self.widmap.meta)

            if self.filename_data is not None:
                self.data = eispac.read_cube(self.filename_data, window=self.fit.meta["iwin"])
            else:
                try:
                    self.filename_data = os.path.join(os.path.dirname(self.filename_fit),
                                                        os.path.basename(self.filename_fit).split(".")[0] + ".data.h5")
                    self.data = eispac.read_cube(self.filename_data, window=self.fit.meta["iwin"])
                except:
                    try:
                        self.filename_data = self.fit.meta["filename_data"]
                        self.data = eispac.read_cube(self.filename_data, window=self.fit.meta["iwin"])
                    except:
                        raise FileNotFoundError("Cannot find the corresponding data file suggested by the fit.h5 or in the same directory."
                                                " Please provide the data file.")
            
            self.fit_wave_range = self.fit.fit["wave_range"]
            fit_wave_range_len = self.fit_wave_range[1] - self.fit_wave_range[0]
            self.select_wave_range = np.array([self.fit_wave_range[0] - 0.2*fit_wave_range_len, self.fit_wave_range[1] + 0.2*fit_wave_range_len])
            # point1 = [SpectralCoord(select_wave_range[0], unit=u.AA), None]
            # point2 = [SpectralCoord(select_wave_range[1], unit=u.AA), None]
            # self.data = self.data.crop(point1, point2)


    def _init_gui(self):
        '''
        Initialize the GUI.
        '''
        if self.data_type == "data":
            self.fig = plt.figure(figsize=(self.xy_ratio*6 + 5, 5*1.1), layout='constrained')
            self.fig1, self.fig2 = self.fig.subfigures(1,2,width_ratios=[self.xy_ratio*1.3, 1],)

            self.ax1 = self.fig1.add_subplot(projection=self.intmap)
            self.intmap.plot(axes=self.ax1, aspect=self.data.meta["aspect"], 
                             norm=ImageNormalize(vmin=np.max((0, np.nanpercentile(self.intmap.data, 1))),
                                                 vmax=np.nanpercentile(self.intmap.data, 99),
                                                 stretch=AsinhStretch()),
                             cmap="plasma",
                             title=None)
            
            self.ax1.set_title(self.data.meta["wininfo"][self.data.meta["iwin"]][1],
                               fontsize=10)
            
            self.ax_line_profile = self.fig2.subplots(sharex=True)
            self.ax_line_profile.set_xlabel(r"Wavelength [$\rm \AA$]")
            
            
        elif self.data_type == "fit":
            self.fig = plt.figure(figsize=(self.xy_ratio*6*3 + 5, 5*1.1), layout='constrained')
            self.fig1, self.fig2 = self.fig.subfigures(1,2,width_ratios=[self.xy_ratio*1.3*3, 1])
            self.ax1 = self.fig1.add_subplot(131,projection=self.intmap)
            im1 = self.intmap.plot(axes=self.ax1, aspect=self.data.meta["aspect"], 
                             norm=ImageNormalize(vmin=np.max((0, np.nanpercentile(self.intmap.data, 1))),
                                                 vmax=np.nanpercentile(self.intmap.data, 99),
                                                 stretch=AsinhStretch()),
                             cmap="plasma",
                             title=None)
            
            
            clb1, clb_ax1 = plot_colorbar(im1, self.ax1, bbox_to_anchor=(0.15, 1.02, 0.7, 0.1*self.xy_ratio),fontsize=10,
                                            orientation="horizontal",
                                            title=None,
                                            scilimits=(-2,2))
            self.fig.canvas.draw()

            clb_ax1.xaxis.tick_top()
            # clb_ax1.tick_params(axis="x", which="both", bottom=False, top=True, labelbottom=False, labeltop=True)
            clb_ax1.xaxis.set_label_position("top")
            if clb_ax1.xaxis.get_offset_text().get_text() != "":
                # clb_ax1.xaxis.get_offset_text().set_position((1.1, 0))
                clb_ax1.xaxis._update_offset_text_position = types.MethodType(top_offset, clb_ax1.xaxis)

            clb_ax1.set_xlabel(r"Intensity [$\rm erg\,s^{-1}\,cm^{-2}\,sr^{-1}$]", fontsize=10)
            
            self.ax2 = self.fig1.add_subplot(132, projection=self.velmap)
            self.ax2.sharex(self.ax1)
            self.ax2.sharey(self.ax1)

            vel_lim = np.max(np.abs([np.nanpercentile(self.velmap.data, 3), np.nanpercentile(self.velmap.data, 97)]))
            im2 = self.velmap.plot(axes=self.ax2, aspect=self.data.meta["aspect"], 
                             norm=ImageNormalize(vmin=-vel_lim, vmax=vel_lim),
                             cmap="coolwarm",
                             title=None)
            
            clb2, clb_ax2 = plot_colorbar(im2, self.ax2, bbox_to_anchor=(0.15, 1.02, 0.7, 0.1*self.xy_ratio),fontsize=10,
                                            orientation="horizontal",
                                            title=None,
                                            scilimits=(-2,2))
            
            clb_ax2.xaxis.tick_top()
            clb_ax2.xaxis.set_label_position("top")
            clb_ax2.set_xlabel(r"Vlos [$\rm km\,s^{-1}$]", fontsize=10)
            
            self.ax3 = self.fig1.add_subplot(133, projection=self.widmap)
            self.ax3.sharex(self.ax1)
            self.ax3.sharey(self.ax1)

            im3 = self.widmap.plot(axes=self.ax3, aspect=self.data.meta["aspect"],
                                norm=ImageNormalize(vmin=np.nanpercentile(self.widmap.data, 1),
                                                    vmax=np.nanpercentile(self.widmap.data, 99)),
                                cmap="cividis",
                                title=None)
            
            clb3, clb_ax3 = plot_colorbar(im3, self.ax3, bbox_to_anchor=(0.15, 1.02, 0.7, 0.1*self.xy_ratio),fontsize=10,
                                            orientation="horizontal",
                                            title=None,
                                            scilimits=(-2,2))
            
            clb_ax3.xaxis.tick_top()
            clb_ax3.xaxis.set_label_position("top")
            clb_ax3.set_xlabel(r"Veff [$\rm km\,s^{-1}$]", fontsize=10)
            
            self.ax_line_profile, self.ax_residual = self.fig2.subplots(2,1,sharex=True,gridspec_kw={"height_ratios":[4,1],
                                                                                        "top":0.95,
                                                                                        "bottom":0.05})
            
            self.ax_line_profile.set_title(self.intmap.meta["line_id"] + "\n ", fontsize=10)

            for ax_ in (self.ax2, self.ax3):
                ax_.coords[0].axislabels.set_visible(False)
                ax_.coords[1].axislabels.set_visible(False)
                ax_.coords[1].set_ticklabel_visible(False)

            self.ax_residual.set_xlabel(r"Wavelength [$\rm \AA$]")
            self.ax_residual.set_ylabel("Residual")
            self.ax_residual.set_yticks([])

            for ax_ in (self.ax_line_profile, self.ax_residual):
                ax_.tick_params(direction="in")
            
            self.ax_line_profile.set_xlim(*self.select_wave_range)
        
        self.ax1.coords[0].set_axislabel("Solar-X [arcsec]",fontsize=10)
        self.ax1.coords[1].set_axislabel("Solar-Y [arcsec]",fontsize=10)

        self.line_data = self.ax_line_profile.errorbar(np.nanmean(self.data.wavelength, axis=(0,1)),
                                                        np.nanmean(self.data.data, axis=(0,1)),
                                                        yerr=np.nanstd(self.data.data, axis=(0,1)),
                                                        ds='steps-mid', capsize=2, lw=1.5, color="#E87A90")
        self.line_data_fill = self.ax_line_profile.fill_between(np.nanmean(self.data.wavelength, axis=(0,1)),
                                                                    np.ones(self.data.data.shape[-1])*np.min([-0.1*np.max(np.nanmean(self.data.data, axis=(0,1))),0]),
                                                                    np.nanmean(self.data.data, axis=(0,1)),
                    step='mid',color="#FEDFE1",alpha=0.6)
        
        self.line_all_fit = None
        self.line_single_fits = None
        
        # update_errorbar_visibility(self.line_data, False)
        # self.line_data_fill.set_visible(False)
        
        
        self.ax_line_profile.set_ylabel(r"Intensity [$\rm erg\,s^{-1}\,cm^{-2}\,sr^{-1}$]")
        self.ax_line_profile.ticklabel_format(axis="y", style="sci", scilimits=(-2,2))

        

        self.select_x = None
        self.select_y = None
        self.skycoord = None
        self.markers = None

        self.fig.get_layout_engine().set(w_pad=4/72, h_pad=0/72, hspace=0,
                            wspace=0,rect=[0,0.02,1,0.96])
        # self.fig1.get_layout_engine().set(w_pad=2/72, h_pad=4/72, hspace=0,
        #                     wspace=0,)
        # self.fig2.get_layout_engine().set(w_pad=2/72, h_pad=0/72, hspace=0,
        #                     wspace=0,)
        
        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # print window info
        print("{:>4s} {:>16s} {:>8s} {:>8s}".format("Idx", "Win Name", "wmin", "wmax"))
        for row_ in self.data.meta["wininfo"]:
            if self.data.meta["iwin"] == row_[0]:
                print("{:4d} {:>16s} {:>8.3f} {:>8.3f} <---".format(*row_))
            else:
                print("{:4d} {:>16s} {:>8.3f} {:>8.3f}".format(*row_))

        plt.show()

    def _on_click(self, event):
        '''
        The event handler when clicking on the map.
        '''
        if self.fig.canvas.toolbar.mode not in ["pan/zoom", "zoom rect"]:
            if self.data_type == "data":
                if event.inaxes == self.ax1:
                    self.select_x = np.round(event.xdata).astype(int)
                    self.select_y = np.round(event.ydata).astype(int)

                    self.skycoord = self.intmap.pixel_to_world(self.select_x*u.pix, self.select_y*u.pix)
                    self.ax_line_profile.set_title(f"({self.select_x:.0f}, {self.select_y:.0f}) " 
                                            f"({self.skycoord.Tx.to_value(u.arcsec):.1f}\", {self.skycoord.Ty.to_value(u.arcsec):.1f}\")",
                                            fontsize=10)

                    self.update_line_profile()
            elif self.data_type == "fit":
                if event.inaxes in (self.ax1, self.ax2, self.ax3):
                    self.select_x = np.round(event.xdata).astype(int)
                    self.select_y = np.round(event.ydata).astype(int)

                    self.skycoord = self.intmap.pixel_to_world(self.select_x*u.pix, self.select_y*u.pix)
                    self.ax_line_profile.set_title(self.intmap.meta["line_id"] + "\n"
                                            f"({self.select_x:.0f}, {self.select_y:.0f}) " 
                                            f"({self.skycoord.Tx.to_value(u.arcsec):.1f}\", {self.skycoord.Ty.to_value(u.arcsec):.1f}\")",
                                            fontsize=10)

                    self.update_line_profile()

        
    def update_line_profile(self):
        '''
        Update the line profile plot according to the selected pixel.
        '''
        
        update_errorbar_visibility(self.line_data, True)
        update_errorbar(self.line_data, self.data.wavelength[self.select_y, self.select_x, :],
                        self.data.data[self.select_y, self.select_x, :],
                        yerr=np.abs(self.data.uncertainty.array[self.select_y, self.select_x, :]))

        self.line_data_fill.set_visible(True)
        update_fill_between(self.line_data_fill, self.data.wavelength[self.select_y, self.select_x, :],
                            np.ones(self.data.data.shape[-1])*np.min([-0.1*np.max(self.data.data[self.select_y, self.select_x, :]), 0]),
                            self.data.data[self.select_y, self.select_x, :])
        
        if self.data_type == "fit":
            if self.line_all_fit is None:
                fit_x, fit_y = self.fit.get_fit_profile(coords=[self.select_y,self.select_x], num_wavelengths=100)
                self.line_all_fit, = self.ax_line_profile.plot(fit_x, fit_y, color="black", lw=1.5)

                fit_x_res, fit_y_res = self.fit.get_fit_profile(coords=[self.select_y,self.select_x],)
                fit_y_res = self.data.data[self.select_y, self.select_x, :] - fit_y_res
                fit_x_res = fit_x_res.filled(np.nan)
                fit_y_res = fit_y_res.filled(np.nan)

                self.line_residual = self.ax_residual.errorbar(fit_x_res, fit_y_res, yerr=np.abs(self.data.uncertainty.array[self.select_y, self.select_x, :]),
                                                                ds='steps-mid', capsize=2, lw=1.5, color="#E87A90")
                self.ax_residual.axhline(0, color="grey", lw=1.5, ls="--")

                if self.fit.n_gauss > 1:
                    self.line_single_fits = []
                    fit_x, fit_bg = self.fit.get_fit_profile(coords=[self.select_y,self.select_x], num_wavelengths=100, component=self.fit.n_gauss)
                    for ii in range(self.fit.n_gauss):
                        fit_x, fit_y = self.fit.get_fit_profile(coords=[self.select_y,self.select_x], num_wavelengths=100, component=ii)
                        line_, = self.ax_line_profile.plot(fit_x,fit_y + fit_bg,color="#E9002D",
                                                           ls="-" if ii == self.index else "--",
                                                           lw=1.5,alpha=0.7)
                        self.line_single_fits.append(line_)

                self.text = self.ax_line_profile.text(0.05, 0.95, f"Int: {self.intmap.data[self.select_y, self.select_x]:.2e}\n"
                                                                f"Vel: {self.velmap.data[self.select_y, self.select_x]:.2f} km/s\n"
                                                                f"Wid: {self.widmap.data[self.select_y, self.select_x]:.2f} km/s\n"
                                                                f"Chi2: {self.fit.fit["chi2"][self.select_y, self.select_x]:.2f}",
                                                                transform=self.ax_line_profile.transAxes,
                                                                fontsize=10, ha="left", va="top", color="black",
                                                                linespacing=1.3)
                
                # self.ax_residual.grid(True)
                # self.ax_line_profile.grid(True)

            else:
                fit_x, fit_y = self.fit.get_fit_profile(coords=[self.select_y,self.select_x], num_wavelengths=100)
                self.line_all_fit.set_data(fit_x, fit_y)

                if self.fit.n_gauss > 1:
                    fit_x, fit_bg = self.fit.get_fit_profile(coords=[self.select_y,self.select_x], num_wavelengths=100, component=self.fit.n_gauss)
                    for ii, line_ in enumerate(self.line_single_fits):
                        fit_x, fit_y = self.fit.get_fit_profile(coords=[self.select_y,self.select_x], num_wavelengths=100, component=ii)
                        line_.set_data(fit_x, fit_y + fit_bg)
                
                fit_x_res, fit_y_res = self.fit.get_fit_profile(coords=[self.select_y,self.select_x],)
                fit_y_res = self.data.data[self.select_y, self.select_x, :] - fit_y_res
                fit_x_res = fit_x_res.filled(np.nan)
                fit_y_res = fit_y_res.filled(np.nan)

                update_errorbar(self.line_residual, fit_x_res, fit_y_res, yerr=np.abs(self.data.uncertainty.array[self.select_y, self.select_x, :]))
                
                self.text.set_text(f"Int: {self.intmap.data[self.select_y, self.select_x]:.2e}\n"
                                f"Vel: {self.velmap.data[self.select_y, self.select_x]:.2f} km/s\n"
                                f"Wid: {self.widmap.data[self.select_y, self.select_x]:.2f} km/s\n"
                                f"Chi2: {self.fit.fit["chi2"][self.select_y, self.select_x]:.2f}")
                
                self.ax_residual.relim()
                self.ax_residual.autoscale(axis="y")
        
        self.ax_line_profile.relim()
        self.ax_line_profile.autoscale(axis="y")
        self.ax_line_profile.set_ylim(bottom=np.min([-0.1*np.max(self.data.data[self.select_y, self.select_x, :]),0]))

        if self.markers is None:

            if self.data_type == "data":
                self.markers = [mlines.Line2D([self.select_x], [self.select_y], marker='X', markerfacecolor='#080808', markersize=8,
                                            linewidth=5, markeredgecolor='white', markeredgewidth=1)]
                self.ax1.add_line(self.markers[0])
            elif self.data_type == "fit":
                self.markers = []
                for ax_ in (self.ax1, self.ax2, self.ax3):
                    marker_ = mlines.Line2D([self.select_x], [self.select_y], marker='X', markerfacecolor='#080808', markersize=8,
                                            linewidth=5, markeredgecolor='white', markeredgewidth=1)
                    self.markers.append(ax_.add_line(marker_))
        else:
            for marker in self.markers:
                marker.set_xdata([self.select_x])
                marker.set_ydata([self.select_y])
            

        self.fig.canvas.draw_idle()
            
    
def plot_colorbar(im, ax, bbox_to_anchor=(1.02, 0., 1, 0.1),fontsize=10,
                  orientation="vertical",
                  title=None,scilimits=(-4,4),**kwargs):
    '''
    Plot a colorbar with the specified orientation and title.
    '''

    clb_ax = ax.inset_axes(bbox_to_anchor,transform=ax.transAxes)
    
    clb = plt.colorbar(im,pad = 0.05,orientation=orientation,ax=ax,cax=clb_ax,**kwargs)
    
    clb_ax.tick_params(labelsize=fontsize)
    
    if orientation == "vertical":
        clb_ax.ticklabel_format(axis="y", style="sci", scilimits=scilimits)
        clb_ax.yaxis.get_offset_text().set_fontsize(fontsize)
        clb_ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    elif orientation == "horizontal":
        clb_ax.ticklabel_format(axis="x", style="sci", scilimits=scilimits)
        clb_ax.xaxis.get_offset_text().set_fontsize(fontsize)
        clb_ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if title is not None:
        clb.set_label(title,fontsize=fontsize)

    return clb, clb_ax

def update_errorbar_visibility(errobj, visible):
    '''
    Update the visibility of the errorbar object.
    '''
    ln, caps, bars = errobj

    ln.set_visible(visible)
    for cap in caps:
        cap.set_visible(visible)
    for bar in bars:
        bar.set_visible(visible)

def update_errorbar(errobj, x, y, xerr=None, yerr=None):
    '''
    Update the errorbar object with new data.
    '''
    ln, caps, bars = errobj


    if len(bars) == 2:
        assert xerr is not None and yerr is not None, "Your errorbar object has 2 dimension of error bars defined. You must provide xerr and yerr."
        barsx, barsy = bars  # bars always exist (?)
        try:  # caps are optional
            errx_top, errx_bot, erry_top, erry_bot = caps
        except ValueError:  # in case there is no caps
            pass

    elif len(bars) == 1:
        assert (xerr is     None and yerr is not None) or\
               (xerr is not None and yerr is     None),  \
               "Your errorbar object has 1 dimension of error bars defined. You must provide xerr or yerr."

        if xerr is not None:
            barsx, = bars  # bars always exist (?)
            try:
                errx_top, errx_bot = caps
            except ValueError:  # in case there is no caps
                pass
        else:
            barsy, = bars  # bars always exist (?)
            try:
                erry_top, erry_bot = caps
            except ValueError:  # in case there is no caps
                pass

    ln.set_data(x,y)

    try:
        errx_top.set_xdata(x + xerr)
        errx_bot.set_xdata(x - xerr)
        errx_top.set_ydata(y)
        errx_bot.set_ydata(y)
    except NameError:
        pass
    try:
        barsx.set_segments([np.array([[xt, y], [xb, y]]) for xt, xb, y in zip(x + xerr, x - xerr, y)])
    except NameError:
        pass

    try:
        erry_top.set_xdata(x)
        erry_bot.set_xdata(x)
        erry_top.set_ydata(y + yerr)
        erry_bot.set_ydata(y - yerr)
    except NameError:
        pass
    try:
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y + yerr, y - yerr)])
    except NameError:
        pass

def vertices_between(x, y1, y2):
    '''
    Calculate the new vertices used to fill the area between two lines.
    '''
    if isinstance(y2, float | int):
        y2 = np.full(x.size, y2)
    x, y1, y2 = cbook.pts_to_midstep(x, y1, y2) 
    new_x = np.hstack((x, x[::-1]))
    new_y = np.hstack((y1, y2[::-1]))
    return np.vstack((new_x, new_y)).T

def update_fill_between(fill_obj, x, y1, y2):
    '''
    Update the filled area between two lines.
    '''
    fill_obj.set_verts([vertices_between(x, y1, y2)])


def top_offset(self, bboxes, bboxes2):
    '''
    Set the offset text position to the top.
    '''
    pad = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]
    top = self.axes.bbox.ymax
    self.offsetText.set(va="bottom", ha="left") 
    oy = top + pad * self.figure.dpi / 72.0
    self.offsetText.set_position((1.02, oy))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect EIS data or fit file.")
    parser.add_argument("-d","--data", type=str, help="The data file to inspect. If only a data file is provided, "
                                                      "the GUI will show the integrated intensity map. "
                                                      "If both data and fit files are provided, the GUI will show the" 
                                                      "fitted intensity, velocity, and width maps.")
    
    parser.add_argument("-f","--fit", type=str, help="The fit file to inspect. If only a fit file is provided, "
                                                     "the routine will search for the corresponding data file in the same directory, "
                                                     "or the file path restored in the fit file. Otherwise, you must provide the data file.")
    
    parser.add_argument("-i","--index", type=int, default=0, help="The index of the window/wavelength (data file) "
                        "or component (fitting file) to inspect.")
    
    args = parser.parse_args()

    eis_inspector = EISInspector(filename_data=args.data, filename_fit=args.fit, index=args.index)
    