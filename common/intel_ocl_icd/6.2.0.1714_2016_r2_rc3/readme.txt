================================================================================
				Intel(R) SDK for OpenCL(tm) Applications 2014 B1
									README
================================================================================

In this document the # sign represents a Linux* shell prompt; text following
this string on the same line represents commands to be executed in Linux* shell.

================================================================================
Table of Contents
================================================================================

1. Introduction
2. Installation Prerequisites
3. Package Description
4. Installing the Product
	4.1 Installing the Product Using Shell Scripts
	4.2 Installing the Product Using Package Manager
5. Uninstalling the Product
7. Product Directories
8. Other Intel Products
9. Disclaimer and Legal Information

================================================================================
1. Introduction
================================================================================

OpenCL(tm) (Open Computing Language) is an open standard for general-purpose
parallel programming.
OpenCL provides a uniform programming environment for writing portable code for
client computer systems, high-performance computing servers, and other compute
systems. OpenCL is developed by multiple companies through the Khronos* OpenCL
committee, and Intel is a key contributor to the OpenCL standard since
inception.

This package contains Intel implementation of OpenCL standard.

Download the relevant Intel SDK for OpenCL Applications package through
the product web site, available at www.intel.com/software/opencl.

================================================================================
2. Installation Prerequisites
================================================================================

For information on supported operating systems and hardware refer to the product
Release Notes.

================================================================================
3. Package Description
================================================================================

Intel SDK for OpenCL Applications 2014 provides the following TGZ archives
for product installation:
	
	- Intel SDK for OpenCL Applications 2014 - runtime only package, which
	contains the OpenCL runtime and compiler for Ubuntu* OS. The TGZ archive
	comprises the following packages:
		- opencl-1.2-base.deb
		- opencl-1.2-intel-cpu.deb
	
	- Intel SDK for OpenCL Applications 2014 - the SDK package, which
	contains the OpenCL runtime, compiler, OpenCL C header files, development
	tools and the OpenCL Installable Client Driver (ICD). The TGZ archive
	comprises the following packages:
		- opencl-1.2-base.deb
		- opencl-1.2-intel-cpu.deb
		- opencl-1.2-devel.deb
		- opencl-1.2-intel-devel.deb
		- opencl-1.2-intel-devel-android.deb

See the list below for information on DEB packages contents:

	- opencl-1.2-base.deb - contains the OpenCL Installable Client Driver ICD)
	loader. The OpenCL ICD enables different OpenCL implementations to coexist
	on the same system. ICD also enables applications to select between OpenCL
	implementations at run time.
	
	- opencl-1.2-intel-cpu.deb - contains the OpenCL runtime and compiler,
	which enable Intel CPU as an OpenCL device. OpenCL 1.2 features are
	supported on all generations of the Intel Core(tm) processors and the
	Intel Xeon processors.
	
	- opencl-1.2-devel.deb - contains OpenCL C header files, which enable
	development of OpenCL applications on your machine.
	
	- opencl-1.2-intel-devel.deb - contains the following Intel SDK for
	OpenCL Applications development tools:
		- Intel SDK for OpenCL - Kernel Builder, which enables building and
		analyzing OpenCL kernels and provides full offline OpenCL language
		compilation.
		- Intel SDK for OpenCL - Offline Compiler, a command-line utility,
		which enables offline compilation and building of OpenCL kernels.
	
	- opencl-1.2-intel-devel-android.deb - contains the following components:
		- script for OpenCL runtime installation on Android devices.
		- Eclipse* IDE plugin, which enables OpenCL compilation.

================================================================================
4. Installing the Product
================================================================================

To proceed to installation, extract the relevant TGZ archive contents:

	# tar xzf intel_sdk_for_ocl_applications_2014_beta_sdk_4.0.*_x64.tgz
	# cd intel_sdk_for_ocl_applications_2014_beta_sdk_4.0.*_x64

--------------------------------------------------------------------------------
4.1 Installing the Product Using Shell Scripts
--------------------------------------------------------------------------------

To install the product, run the following command:
	# ./install-cpu.sh

________________________________________________________________________________
NOTE: Installing the CPU runtime using scripts without prior uninstallation of 
the CPU-only runtime is not supported.
________________________________________________________________________________

--------------------------------------------------------------------------------
4.2 Installing the Product Using Package Manager
--------------------------------------------------------------------------------

To install the CPU-only runtime, run the following command:

	# sudo apt-get install *base*.deb *intel-cpu*.deb

To install the development tools and runtime, run the following commands:

	# sudo apt-get install *base*.deb *intel-cpu*.deb *devel*.deb
________________________________________________________________________________
NOTE: In case of SDK installation on a machine without runtime components, the
CPU-only runtime is installed together with the tool set.
________________________________________________________________________________


================================================================================
5. Uninstalling the Product
================================================================================

To uninstall the product using the uninstallation script, do the following:
    1.  Go to the folder to which you extracted the TGZ archive content.
    2.  Run the script: 

	# ./uninstall.sh
________________________________________________________________________________
NOTE: The uninstall.sh script erases all packages that match "opencl-1.2".
________________________________________________________________________________

To remove all packages, starting with 'opencl-1.2-', run the following commands:

	# sudo apt-get remove "opencl-1.2-*"

================================================================================
6. Product Directories
================================================================================

The following directory map indicates the default structure of the installed
files and identifies the contents of the main sub-directories. Availability of
directories depends on the installed DEB packages.

    +-- opt
        |
        +-- intel
            |
            +-- opencl
                |
                +-- android-preinstall (Android runtime and installation script)
		|
                +-- bin (executable files)
                |
                +-- doc (documentation files)
                |
                +-- eclipse-plug-in (JAR file)
		|
                +-- etc (ICD file)
                |
                +-- include (OpenCL C header files)
                |
                +-- lib64 (OpenCL libraries)

================================================================================
7. Other Intel Products
================================================================================
You can find out about other Intel software development products through the
Intel web site at: http://www.intel.com/software/products/

================================================================================
8. Disclaimer and Legal Information
================================================================================
INFORMATION IN THIS DOCUMENT IS PROVIDED IN CONNECTION WITH INTEL PRODUCTS.
NO LICENSE, EXPRESS OR IMPLIED, BY ESTOPPEL OR OTHERWISE, TO ANY INTELLECTUAL
PROPERTY RIGHTS IS GRANTED BY THIS DOCUMENT. EXCEPT AS PROVIDED IN INTEL'S
TERMS AND CONDITIONS OF SALE FOR SUCH PRODUCTS, INTEL ASSUMES NO LIABILITY
WHATSOEVER AND INTEL DISCLAIMS ANY EXPRESS OR IMPLIED WARRANTY, RELATING TO
SALE AND/OR USE OF INTEL PRODUCTS INCLUDING LIABILITY OR WARRANTIES RELATING TO
FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABILITY, OR INFRINGEMENT OF ANY
PATENT, COPYRIGHT OR OTHER INTELLECTUAL PROPERTY RIGHT.
A "Mission Critical Application" is any application in which failure of the
Intel Product could result, directly or indirectly, in personal injury or death.
SHOULD YOU PURCHASE OR USE INTEL'S PRODUCTS FOR ANY SUCH MISSION CRITICAL
APPLICATION, YOU SHALL INDEMNIFY AND HOLD INTEL AND ITS SUBSIDIARIES,
SUBCONTRACTORS AND AFFILIATES, AND THE DIRECTORS, OFFICERS, AND EMPLOYEES OF
EACH, HARMLESS AGAINST ALL CLAIMS COSTS, DAMAGES, AND EXPENSES AND REASONABLE
ATTORNEYS' FEES ARISING OUT OF, DIRECTLY OR INDIRECTLY, ANY CLAIM OF PRODUCT
LIABILITY, PERSONAL INJURY, OR DEATH ARISING IN ANY WAY OUT OF SUCH MISSION
CRITICAL APPLICATION, WHETHER OR NOT INTEL OR ITS SUBCONTRACTOR WAS NEGLIGENT
IN THE DESIGN, MANUFACTURE, OR WARNING OF THE INTEL PRODUCT OR ANY OF ITS
PARTS.
Intel may make changes to specifications and product descriptions at any time,
without notice. Designers must not rely on the absence or characteristics of
any features or instructions marked "reserved" or "undefined". Intel reserves
these for future definition and shall have no responsibility whatsoever for
conflicts or incompatibilities arising from future changes to them. The
information here is subject to change without notice. Do not finalize a
design with this information. The products described in this document may
contain design defects or errors known as errata which may cause the product
to deviate from published specifications. Current characterized errata are
available on request. Contact your local Intel sales office or your
distributor to obtain the latest specifications and before placing your
product order.
Copies of documents which have an order number and are referenced in this
document, or other Intel literature, may be obtained by calling 1-800-548-4725,
or go to:
http://www.intel.com/design/literature.htm.
Intel processor numbers are not a measure of performance.  Processor numbers
differentiate features within each processor family, not across different
processor families. Go to: http://www.intel.com/products/processor_number/.
Software and workloads used in performance tests may have been optimized for
performance only on Intel microprocessors.  Performance tests, such as SYSmark
and MobileMark, are measured using specific computer systems, components,
software, operations and functions.  Any change to any of those factors may
cause the results to vary.  You should consult other information and
performance tests to assist you in fully evaluating your contemplated
purchases, including the performance of that product when combined with
other products.
This document contains information on products in the design phase of 
development.
Intel, Intel logo, Intel Core, VTune, Xeon, Xeon Phi are trademarks of
Intel Corporation in the U.S. and other countries.
* Other names and brands may be claimed as the property of others.
OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission
from Khronos.
Copyright © 2011-2016 Intel Corporation. All rights reserved.
