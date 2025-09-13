###------------R seurat workflow-------------
#-----1.load data-----
read_data <- function(myname, mydir){
  sample_list <- list()
  
  for (i in 1:length(myname)) {
    sampleID <- myname[i]
    
    mydir <- as.character(mydir)
    TenXdat <- Read10X(data.dir = paste0(mydir, '/', sampleID, "/outs/filtered_feature_bc_matrix/"))
    TenXdat <- CreateSeuratObject(counts = TenXdat, min.cells = 3, min.features = 200, project = '')
    
    mito.features <- grep(pattern = "^MT-", x = rownames(x = TenXdat), value = TRUE)
    TenXdat[["percent.mt"]] <- PercentageFeatureSet(TenXdat, pattern = "MT-")
    
    TenXdat <- subset(x = TenXdat, subset = nFeature_RNA >= 750 & percent.mt <= 5 & nCount_RNA >= 1000)
    
    sample_list[[sampleID]] <- TenXdat
  }
  return(sample_list)
}


#-----2.QC------
qc_process <- function(merged_data){
  pbmc_list <- list()  
  for (i in 1:(table(merged_data$pbmc_sample) %>% length)) {
    pbmc <- table(merged_data$pbmc_sample) %>% names()
    pbmcID <- pbmc[i]
    
    TenXdat <- merged_data[, merged_data$pbmc_sample %in% pbmcID]
    
    mito.features <- grep(pattern = "^MT-", x = rownames(x = TenXdat), value = TRUE)
    TenXdat[["percent.mt"]] <- PercentageFeatureSet(TenXdat, pattern = "^MT-")
    
    TenXdat <- subset(x = TenXdat, subset = nFeature_RNA >= 750 & percent.mt <= 5 & nCount_RNA >= 1000)
    TenXdat <- NormalizeData(object = TenXdat, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
    TenXdat <- FindVariableFeatures(object = TenXdat, nfeatures = 2000, verbose = FALSE)
    TenXdat <- ScaleData(object = TenXdat, features = VariableFeatures(object = TenXdat), vars.to.regress = c("nCount_RNA", "percent.mt"), verbose = FALSE)
    TenXdat <- RunPCA(object = TenXdat, features = VariableFeatures(object = TenXdat), verbose = FALSE)
    
    subset_cells <- TenXdat
    
    dim.use <- 30
    res.use <- 1
    
    subset_cells <- FindNeighbors(object = subset_cells, dims = 1:dim.use, verbose = FALSE)
    subset_cells <- FindClusters(object = subset_cells, resolution = res.use, verbose = FALSE)
    subset_cells <- RunUMAP(object = subset_cells, dims = 1:dim.use, umap.method = "uwot", verbose = FALSE)
    
    
    ##--------------------step1. SoupX flow--------------------
    sc <- SoupChannel(subset_cells@assays$RNA@counts, 
                      subset_cells@assays$RNA@counts, calcSoupProfile = FALSE)
    
    toc <- sc$toc
    soupProf <- data.frame(row.names = rownames(toc), 
                           est = rowSums(data.frame(toc))/sum(toc), 
                           counts = rowSums(data.frame(toc)))
    
    sc <- setSoupProfile(sc, soupProf)

    sc <- setClusters(sc, subset_cells$seurat_clusters %>% as.character %>% as.numeric())
    
    sc <- autoEstCont(sc, forceAccept=TRUE)
    out <- adjustCounts(sc)
    subset_cells@assays$RNA@counts <- round(out)
    
    
    ##--------------------step2. DoubletFinder flow--------------------
    subset_cells <- subset(subset_cells, subset = nFeature_RNA >= 750 & percent.mt <= 5 & nCount_RNA >= 1000)
    subset_cells <- NormalizeData(object = subset_cells, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
    subset_cells <- FindVariableFeatures(object = subset_cells, nfeatures = 2000, verbose = FALSE)
    subset_cells <- ScaleData(object = subset_cells, features = VariableFeatures(object = subset_cells), vars.to.regress = c("nCount_RNA", "percent.mt"), verbose = FALSE)
    subset_cells <- RunPCA(object = subset_cells, features = VariableFeatures(object = subset_cells), verbose = FALSE)
    subset_cells <- FindNeighbors(object = subset_cells, dims = 1:dim.use, verbose = FALSE)
    subset_cells <- FindClusters(object = subset_cells, resolution = res.use, verbose = FALSE)
    subset_cells <- RunUMAP(object = subset_cells, dims = 1:dim.use, umap.method = "uwot", verbose = FALSE)
    
    ##find best pK
    sweep.res.list <- paramSweep_v3(subset_cells, PCs = subset_cells@commands$FindNeighbors.RNA.pca$dims)
    sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)
    bcmvn <- find.pK(sweep.stats)
    pK <- bcmvn[which(max(bcmvn$BCmetric) == bcmvn$BCmetric),2] %>% as.character %>% as.numeric
    
    homotypic.prop <- modelHomotypic(subset_cells@active.ident)
    nExp_poi <- round(0.07*ncol(subset_cells))
    nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))
    
    ##--------------identify doublets---------------
    subset_cells <- doubletFinder_v3(subset_cells, PCs = subset_cells@commands$FindNeighbors.RNA.pca$dims, pN = 0.25, 
                                     pK = pK, nExp = nExp_poi, reuse.pANN = FALSE)
    subset_cells <- doubletFinder_v3(subset_cells, PCs = subset_cells@commands$FindNeighbors.RNA.pca$dims, pN = 0.25, 
                                     pK = pK, nExp = nExp_poi.adj, reuse.pANN = grep("pANN", names(subset_cells@meta.data), value = T))
    
    high_of_low <- (subset_cells@meta.data[, grep("^DF\\.classifications", names(subset_cells@meta.data), value = T)] == "Singlet")+0
    subset_cells@meta.data[high_of_low[, 1] + high_of_low[, 2] == 2, "DF_hi.lo"] <- "Singlet"
    subset_cells@meta.data[high_of_low[, 1] + high_of_low[, 2] == 1, "DF_hi.lo"] <- "Doublet_lo"
    subset_cells@meta.data[high_of_low[, 1] + high_of_low[, 2] == 0, "DF_hi.lo"] <- "Doublet_hi"
    
    ##-----------remove the doublets------------------
    subset_cells <- subset_cells[, subset_cells@meta.data[grepl(subset_cells$DF_hi.lo, pattern = "Singlet"), ] %>% row.names()]
    
    pbmc_list[[pbmcID]] <- subset_cells
    
    print(str_glue('{i} {pbmcID} is done √ √ √'))
  }
  return(pbmc_list)
}



###--------------python scanpy&harmony workflow--------------
import igraph
import numpy as np
import scanpy as sc
import anndata
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
import harmonypy as hm
import seaborn as sns
import scanpy.external as sce
import celltypist
from celltypist import models

cell.raw = cell
cell = cell[:, Variable_gene['x']].copy()
sc.pp.scale(cell, zero_center=True, max_value=10)
sc.pp.pca(cell, svd_solver='randomized')

sce.pp.harmony_integrate(cell, key='Sample')  
sc.pp.neighbors(cell, use_rep='X_pca_harmony')  

sc.tl.umap(cell)
sc.tl.leiden(cell, resolution=2, flavor="igraph")
