"""Twelve-panel recombination composite from frozen summaries and saved features."""
from __future__ import annotations

import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

from dense_style import (
    HERE, BLUE, PURPLE, PINK, GRAY, INK, GRID, axes, figure, label,
    csv_rows, effect, numeric, save, strip,
)

COHORTS = ("adjacent", "graph_first_observed", "raw_first_observed")
TITLES = ("Adjacent", "Graph-first", "Raw-first")
COMPARISONS = [
    ("S", "S+prox", "S → S+P"), ("S", "S+z", "S → S+Z"),
    ("S", "S+rc", "S → S+R"), ("S", "S+prox+rc", "S → S+P+R"),
    ("S", "S+prox+z", "S → S+P+Z"), ("S", "S+prox+z+rc", "S → S+P+Z+R"),
    ("S+prox", "S+prox+rc", "+R after P"),
    ("S+prox+z", "S+prox+z+rc", "+R after P+Z"),
]
COL = (170-4-6)/3
X = (2, 2+COL+3, 2+2*(COL+3))


def one(rows, **criteria):
    found = [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one stored RC comparison: {criteria}; got {len(found)}")
    return found[0]


def draw_geometry(fig, data):
    """Show observed pair density and the actual saved annual regression."""
    distance = data["distance_density"]
    residual = data["residual_density"]
    fit = data["fit"]
    if fit["maximum_identity_error"] > 1e-9:
        raise ValueError("Saved RC identity disagrees with its stored coefficients")
    n = fit["n_defined"]
    if n != distance["n"] or n != residual["n"]:
        raise ValueError("Density and fitted pair populations differ")
    vmax = max(np.log10(np.asarray(d["counts"]).max()+1) for d in [distance,residual])
    cmap = LinearSegmentedColormap.from_list("pair_density", ["#FFFFFF", "#D9E2F1", BLUE, "#183653"])
    label(fig, "a", "Prior distance and network degree", X[0], 203.5)
    label(fig, "b", "Residual geometry", X[2], 203.5)
    axs = [axes(fig, 14, 167, 84, 31), axes(fig, X[2]+9, 167, COL-11, 31)]
    for ax, d, ylabel in zip(axs, [distance,residual], ["1 − P", "R"], strict=True):
        counts = np.asarray(d["counts"])
        if int(counts.sum()) != n:
            raise ValueError("Density dropped a defined-P pair")
        mesh = ax.pcolormesh(d["x_edges"], d["y_edges"], np.log10(counts.T+1),
                             cmap=cmap,vmin=0,vmax=vmax,shading="flat",rasterized=False)
        ax.set(xlabel="Z = log(degree product + 1)", ylabel=ylabel)
        ax.tick_params(pad=1.5)
        ax.set_xticks([9,11,13])
        ax.set_yticks([0,.5,1] if ylabel=="1 − P" else [-.5,0,.5])
    xx = np.asarray(distance["x_edges"])[[0,-1]]
    axs[0].plot(xx,fit["slope"]*xx+fit["intercept"],color=INK,lw=1,ls=(0,(4,2)))
    axs[1].axhline(0,color=INK,lw=.7,ls=(0,(3,2)))
    axs[1].set_xlabel("Z")
    cax = axes(fig, 101, 170, 2, 25)
    cb = fig.colorbar(mesh,cax=cax)
    cb.solids.set_rasterized(False)
    cb.set_label("log₁₀(pairs + 1)",fontsize=8,labelpad=2)
    cb.ax.tick_params(labelsize=8,pad=1,length=2)
    # The annotated key is a prespecified historical example, not a chosen extreme.
    for i, point in enumerate(data["annotations"]):
        text = "\n".join(point["pair_names"])
        axs[0].annotate(text,(point["z"],point["distance"]),xytext=(4,7+13*i),
                        textcoords="offset points",fontsize=8,ha="left",va="bottom",
                        arrowprops={"arrowstyle":"-","color":INK,"lw":.6})
        axs[0].plot(point["z"],point["distance"],marker="D",ms=4,mfc="white",mec=INK,mew=.7)
    strip(fig,f"2016 · DF20 · seed 0 · {n:,} defined-P pairs",14,199)


def draw_populations(fig, cells):
    titles = ("Prior coverage", "Positive fraction", "Baseline S")
    for i, title in enumerate(titles):
        label(fig,chr(ord("c")+i),title,X[i],148.5)
    axcov=axes(fig,23,130,30,15)
    axpos=axes(fig,X[1]+8,130,COL-11,15)
    axap=axes(fig,X[2]+8,130,COL-11,15)
    for y,cohort in zip([2,1,0],COHORTS,strict=True):
        allrow=cells[("all_rows",cohort)][0]
        covered=allrow["n_prox_defined"]/allrow["n_fronts"]*100
        axcov.barh(y,100,height=.6,color="#EAEAEA",edgecolor=GRAY,lw=.35)
        axcov.barh(y,covered,height=.6,color=BLUE,hatch="//",edgecolor=BLUE,lw=.35)
        axcov.text(covered+2,y,f"{covered:.1f}",fontsize=8,va="center")
        for ax,key in [(axpos,"prevalence"),(axap,"ap")]:
            values=[]
            for analysis in ["all_rows","complete_proximity"]:
                row=cells[(analysis,cohort)][0]
                values.append(100*row["n_positive"]/row["n_fronts"] if key=="prevalence" else row["baseline_ap_mean"])
            ax.plot(values,[y,y],color=GRAY,lw=.65,zorder=1)
            ax.plot(values[0],y,marker="o",mfc="white",mec=GRAY,ms=4,mew=.8,ls="none")
            ax.plot(values[1],y,marker="^",mfc=BLUE,mec=BLUE,ms=4,mew=.8,ls="none")
    for ax in [axcov,axpos,axap]:
        ax.set(yticks=[2,1,0],yticklabels=TITLES if ax is axcov else [],ylim=(-.55,2.55))
        numeric(ax)
        ax.tick_params(pad=1)
    axcov.set(xlim=(0,124),xticks=[0,50,100],xlabel="Defined P (%)")
    axpos.set(xlim=(0,12),xticks=[0,5,10],xlabel="Positive (%)")
    axap.set(xlim=(0,.66),xticks=[0,.3,.6],xlabel="AP")
    handles=[Line2D([],[],marker="o",mfc="white",mec=GRAY,ls="none",ms=4,label="All pairs"),
             Line2D([],[],marker="^",color=BLUE,ls="none",ms=4,label="Defined P")]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.68,146.5/210),ncol=2,fontsize=8,
               frameon=False,handlelength=.7,handletextpad=.3,columnspacing=1)


def draw_effects(fig, cells):
    plotted=0;after_pz=[]
    for i,analysis in enumerate(["all_rows","complete_proximity"]):
        y0=83 if i==0 else 37
        title_y=114.5 if i==0 else 68.5
        for j,cohort in enumerate(COHORTS):
            tag=chr(ord("f")+3*i+j)
            title=("All: " if i==0 else "P defined: ")+TITLES[j]
            label(fig,tag,title,X[j],title_y)
            ax=axes(fig,32+47*j,y0,40,28.5)
            cell=cells[(analysis,cohort)]
            for k,(baseline,augmented,_) in enumerate(COMPARISONS):
                row=one(cell,baseline=baseline,augmented=augmented)
                color,marker=(PURPLE,"^") if k==6 else ((PINK,"s") if k==7 else (BLUE,"o"))
                effect(ax,row,7-k,color,marker,filled=k in {0,6,7},size=3.4)
                plotted+=1
                if k==7:after_pz.append(row)
            ax.axhline(1.5,color=GRID,lw=.65)
            ax.set(yticks=range(7,-1,-1),yticklabels=[r[2] for r in COMPARISONS] if j==0 else [],
                   xlim=(-.025,.052),ylim=(-.65,7.65),xticks=[-.02,0,.04],xlabel="ΔAP" if i else None)
            ax.tick_params(axis="y",length=0,pad=1.5)
            ax.tick_params(axis="x",pad=1)
            numeric(ax,signed=True)
    if plotted!=48 or len(after_pz)!=6 or not all(r["conditional_dependence_p2_5"]<0<r["conditional_dependence_p97_5"] for r in after_pz):
        raise ValueError("Incomplete RC matrix or changed R-after-PZ result")
    return plotted


def draw_ecdf(fig, data):
    label(fig,"l","Network position and missing prior vectors",2,23)
    fig.text(107/170,23.5/210,"— Defined P   - - Missing P",fontsize=8,va="bottom")
    count_records=[]
    for j,rec in enumerate(data["degree_ecdf"]):
        if rec["cohort"] != COHORTS[j]:raise ValueError("Unexpected ECDF cohort order")
        ax=axes(fig,X[j]+12,7,COL-15,11.5)
        for name,color,linestyle in [("defined",BLUE,"-"),("missing",GRAY,(0,(3,2)))]:
            d=rec[name];x=np.asarray(d["x"]);cdf=np.asarray(d["cdf"])
            if len(x)!=len(cdf) or not np.all(np.diff(x)>=0) or not np.all(np.diff(cdf)>=0) or not np.isclose(cdf[-1],1):
                raise ValueError("Malformed stored empirical CDF")
            ax.step(np.r_[7.5,x,13.5],np.r_[0,cdf,1],where="post",color=color,ls=linestyle,lw=.8)
        ax.set(xlim=(7.5,13.5),ylim=(0,1.03),xticks=[8,10,12],yticks=[0,1],yticklabels=["0","1"] if j==0 else [])
        ax.tick_params(pad=1)
        ax.text(.02,.85,TITLES[j],transform=ax.transAxes,fontsize=8,va="top")
        if j==0:ax.set_ylabel("ECDF",labelpad=1)
        count_records.append({"cohort":rec["cohort"],"defined":rec["defined"]["n"],"missing":rec["missing"]["n"]})
    return count_records


def main():
    rows=csv_rows("frozen_comparisons.csv")
    data=json.loads((HERE/"domain_cases.json").read_text())["fig4"]
    cells={(a,c):[r for r in rows if r["condition"]==f"rc_df20_{c}" and r["analysis"]==a]
           for a in ["all_rows","complete_proximity"] for c in COHORTS}
    if any(len(cell)!=8 for cell in cells.values()):raise ValueError("Expected six cells of eight RC comparisons")
    fig=figure(210)
    draw_geometry(fig,data);draw_populations(fig,cells);n=draw_effects(fig,cells);ecdf=draw_ecdf(fig,data)
    save(fig,"Fig4","Recombination inputs, saved residual geometry and complete conditional comparisons",{
        "panels":12,"RC_contrasts":n,"RC_cells":6,"contrasts_per_cell":8,
        "R_after_PZ_intervals_spanning_zero":6,"defined_pair_geometry_n":data["fit"]["n_defined"],
        "saved_fit_identity_max_error":data["fit"]["maximum_identity_error"],"ecdf_populations":ecdf,
        "S":"39-column B2+B7","P":"prior cosine proximity","Z":"log(distinct-partner degree product + 1)",
        "R":"saved affine residual of prior distance on Z","required_comparisons":[list(r[:2]) for r in COMPARISONS]})


if __name__=="__main__":main()
