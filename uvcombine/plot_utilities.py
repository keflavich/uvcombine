import numpy as np
from .uvcombine import fftmerge, feather_kernel
from turbustat.statistics import psds
import pylab as pl

def compare_parameters_feather_simple(im, im_hi, im_low, lowresfwhm, pixscale,
                                      suffix="", replacement_threshold=0.5,
                                      psd_axlims=(1e-3,1,10,5e3),
                                      imshow_kwargs={},
                                     ):
    """
    Create diagnostic plots for different simulated feathers
    """

    feathers = {}
    fig1 = pl.figure(1, figsize=(16,12))
    fig2 = pl.figure(2, figsize=(16,12))
    fig3 = pl.figure(3, figsize=(16,12))
    fig1.clf()
    fig2.clf()
    fig3.clf()


    plotnum = 1
    for replace_hires,ls in ((replacement_threshold, '--'),(False,':')):
        for lowpassfilterSD,lw in ((True,4),(False,2)):
            for deconvSD,color in ((True,'r'), (False, 'k')):
                #im_hi = im_interferometered
                #im_low = singledish_im
                lowresscalefactor=1
                highresscalefactor=1

                nax1,nax2 = im.shape
                kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale,)

                fftsum, combo = fftmerge(kfft*1,
                                         ikfft*1,
                                         im_hi*highresscalefactor,
                                         im_low*lowresscalefactor,
                                         replace_hires=replace_hires,
                                         lowpassfilterSD=lowpassfilterSD,
                                         deconvSD=deconvSD,
                                        )
                combo = combo.real
                feathers[replace_hires, lowpassfilterSD, deconvSD] = combo
                resid = im-combo


                pfreq, ppow = psds.pspec(np.fft.fftshift(np.abs(fftsum)))
                name = (("Replace < {}; ".format(replace_hires) if replace_hires else "") +
                        ("filterSD;" if lowpassfilterSD else "")+
                        ("deconvSD" if deconvSD else ""))
                if name == "":
                    name = "CASA defaults"
                pfreq = pfreq[np.isfinite(ppow)]
                ppow = ppow[np.isfinite(ppow)]

                pfreq_resid, ppow_resid = psds.pspec(np.fft.fftshift(np.abs(np.fft.fft2(resid))))
                pfreq_resid = pfreq_resid[np.isfinite(ppow_resid)]
                ppow_resid = ppow_resid[np.isfinite(ppow_resid)]

                ax1 = fig1.add_subplot(3, 3, plotnum)
                ax1.loglog(pfreq, ppow, label=name, linestyle=ls, linewidth=lw, color='k', alpha=0.75)
                ax1.loglog(pfreq_resid, ppow_resid, linestyle=ls, linewidth=lw, color='b', alpha=0.75)
                ax1.axis(psd_axlims)
                ax1.set_title(name)
                ax1.set_xlabel("Frequency")
                ax1.set_ylabel("Power")

                ax2 = fig2.add_subplot(3, 3, plotnum)
                ax2.imshow(combo, interpolation='none', origin='lower', **imshow_kwargs)
                ax2.set_title(name)
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])

                ax3 = fig3.add_subplot(3, 3, plotnum)
                ax3.imshow(resid, interpolation='none', origin='lower', **imshow_kwargs)
                ax3.set_title(name)
                ax3.set_xticklabels([])
                ax3.set_yticklabels([])


                plotnum += 1


    ax1 = fig1.add_subplot(3, 3, plotnum)
    pfreq, ppow = psds.pspec(np.fft.fftshift(np.abs(np.fft.fft2(im))))
    pfreq = pfreq[np.isfinite(ppow)]
    ppow = ppow[np.isfinite(ppow)]
    ax1.loglog(pfreq, ppow, linestyle='-', linewidth=4, color='g', alpha=1)
    ax1.axis(psd_axlims)
    ax1.set_title("Original Image")

    ax2 = fig2.add_subplot(3, 3, plotnum)
    ax2.imshow(im, interpolation='none', origin='lower', **imshow_kwargs)
    ax2.set_title("Original Image")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    ax3 = fig3.add_subplot(3, 3, plotnum)
    ax3.imshow(im, interpolation='none', origin='lower', **imshow_kwargs)
    ax3.set_title("Original Image")
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    fig1.subplots_adjust(hspace=0.1, wspace=0.1)
    fig2.subplots_adjust(hspace=0.01, wspace=0.01)
    fig3.subplots_adjust(hspace=0.01, wspace=0.01)
    for ii in range(5):
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()


    fig1.savefig("parameter_comparison_powerspectra{}.png".format(suffix), bbox_inches='tight')
    fig2.savefig("parameter_comparison_images{}.png".format(suffix), bbox_inches='tight')
    fig3.savefig("parameter_comparison_residuals{}.png".format(suffix), bbox_inches='tight')
    #fig1.legend(loc='best')

    #ax1.set_ylim(1e1,1e5)


def compare_feather_weights(im, im_hi, im_low, lowresfwhm, pixscale,
                            lowres_beam_scales=(0.5, 1, 1.5),
                            suffix="", replacement_threshold=0.5,
                            psd_axlims=(1e-3,1,10,5e3),
                            lowresscalefactor=1,
                            highresscalefactor=1,
                            imshow_kwargs={},
                            feather_kwargs={},
                           ):
    """
    Create diagnostic plots for different simulated feathers
    """

    feathers = {}
    fig1 = pl.figure(1, figsize=(16,12))
    fig2 = pl.figure(2, figsize=(16,12))
    fig3 = pl.figure(3, figsize=(16,12))
    fig1.clf()
    fig2.clf()
    fig3.clf()


    plotnum = 1
    for lowres_beam_scale in lowres_beam_scales:
        #im_hi = im_interferometered
        #im_low = singledish_im

        nax1,nax2 = im.shape
        kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm*lowres_beam_scale, pixscale,)

        fftsum, combo = fftmerge(kfft*1,
                                 ikfft*1,
                                 im_hi*highresscalefactor,
                                 im_low*lowresscalefactor,
                                 **feather_kwargs,
                                )
        combo = combo.real
        feathers[lowres_beam_scale] = combo
        resid = im-combo

        pfreq_hi, ppow_hi = psds.pspec(np.fft.fftshift(np.abs(np.fft.fft2(im_hi))))
        pfreq_hi = pfreq_hi[np.isfinite(ppow_hi)]
        ppow_hi = ppow_hi[np.isfinite(ppow_hi)]

        pfreq_low, ppow_low = psds.pspec(np.fft.fftshift(np.abs(np.fft.fft2(im_low))))
        pfreq_low = pfreq_low[np.isfinite(ppow_low)]
        ppow_low = ppow_low[np.isfinite(ppow_low)]


        pfreq, ppow = psds.pspec(np.fft.fftshift(np.abs(fftsum)))
        name = f"Assumed lowres fwhm = {lowres_beam_scale}$\\times$"

        pfreq = pfreq[np.isfinite(ppow)]
        ppow = ppow[np.isfinite(ppow)]

        pfreq_resid, ppow_resid = psds.pspec(np.fft.fftshift(np.abs(np.fft.fft2(resid))))
        pfreq_resid = pfreq_resid[np.isfinite(ppow_resid)]
        ppow_resid = ppow_resid[np.isfinite(ppow_resid)]

        ax1 = fig1.add_subplot(2, 2, plotnum)
        ax1.loglog(pfreq, ppow, label=name, color='k', alpha=0.75, linewidth=4)
        ax1.loglog(pfreq_resid, ppow_resid, color='b', alpha=0.75, linewidth=4)
        ax1.loglog(pfreq_low, ppow_low, color='purple', alpha=0.5, zorder=-5, linewidth=2)
        ax1.loglog(pfreq_hi, ppow_hi, color='darkred', alpha=0.5, zorder=-5, linewidth=2)
        ax1.axis(psd_axlims)
        ax1.set_title(name)
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Power")

        ax2 = fig2.add_subplot(2, 2, plotnum)
        ax2.imshow(combo, interpolation='none', origin='lower', **imshow_kwargs)
        ax2.set_title(name)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        ax3 = fig3.add_subplot(2, 2, plotnum)
        ax3.imshow(resid, interpolation='none', origin='lower', **imshow_kwargs)
        ax3.set_title(name)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])


        plotnum += 1


    ax1 = fig1.add_subplot(2, 2, plotnum)
    pfreq, ppow = psds.pspec(np.fft.fftshift(np.abs(np.fft.fft2(im))))
    pfreq = pfreq[np.isfinite(ppow)]
    ppow = ppow[np.isfinite(ppow)]
    ax1.loglog(pfreq, ppow, linestyle='-', linewidth=4, color='g', alpha=1)
    ax1.loglog(pfreq_low, ppow_low, color='purple', alpha=0.75, zorder=-5, linewidth=2)
    ax1.loglog(pfreq_hi, ppow_hi, color='darkred', alpha=0.75, zorder=-5, linewidth=2)
    ax1.axis(psd_axlims)
    ax1.set_title("Original Image")

    ax2 = fig2.add_subplot(2, 2, plotnum)
    ax2.imshow(im, interpolation='none', origin='lower', **imshow_kwargs)
    ax2.set_title("Original Image")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    ax3 = fig3.add_subplot(2, 2, plotnum)
    ax3.imshow(im, interpolation='none', origin='lower', **imshow_kwargs)
    ax3.set_title("Original Image")
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    fig1.subplots_adjust(hspace=0.1, wspace=0.1)
    fig2.subplots_adjust(hspace=0.01, wspace=0.01)
    fig3.subplots_adjust(hspace=0.01, wspace=0.01)
    for ii in range(5):
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()


    fig1.savefig("parameter_comparison_lowresfwhm_powerspectra{}.png".format(suffix), bbox_inches='tight')
    fig2.savefig("parameter_comparison_lowresfwhm_images{}.png".format(suffix), bbox_inches='tight')
    fig3.savefig("parameter_comparison_lowresfwhm_residuals{}.png".format(suffix), bbox_inches='tight')
    #fig1.legend(loc='best')

    #ax1.set_ylim(1e1,1e5)
