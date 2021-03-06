################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# This software and the information contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a Non-Disclosure Agreement.  Any reproduction or
# disclosure to any third party without the express written consent of
# NVIDIA is prohibited.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################

USECUFFT = 1
USECUBLAS = 1
#keep = 1
verbose = 1
#maxregisters = 20
SMVERSION_template = 1
#usesm20 = 1

# Add source files here
EXECUTABLE	:= plasticity
# Cuda source files (compiled with cudacc)
CUFILES_sm_13		:= plasticity.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES                :=

################################################################################
# Rules and targets

#ROOTDIR := ../CUDA_WORKSHOP_UIUC/common
#include ../CUDA_WORKSHOP_UIUC/common/common.mk
SHELL = /bin/bash
HOST := $(shell hostname)
ifeq ($(HOST),serenity) 
ROOTDIR := /media/scratch/cuda_trimmed/C/common
else
ROOTDIR := /a/CUDA_SDK_31/C/common
endif

BINDIR := ./build
export $(HEADER)
include common.mk

LIB += -lfftw3
