.. This Software (Dioptra) is being made available as a public service by the
.. National Institute of Standards and Technology (NIST), an Agency of the United
.. States Department of Commerce. This software was developed in part by employees of
.. NIST and in part by NIST contractors. Copyright in portions of this software that
.. were developed by NIST contractors has been licensed or assigned to NIST. Pursuant
.. to Title 17 United States Code Section 105, works of NIST employees are not
.. subject to copyright protection in the United States. However, NIST may hold
.. international copyright in software created by its employees and domestic
.. copyright (or licensing rights) in portions of software that were assigned or
.. licensed to NIST. To the extent that NIST holds copyright in this software, it is
.. being made available under the Creative Commons Attribution 4.0 International
.. license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts
.. of the software developed or licensed by NIST.
..
.. ACCESS THE FULL CC BY 4.0 LICENSE HERE:
.. https://creativecommons.org/licenses/by/4.0/legalcode

What is Dioptra?
================

.. include:: /_glossary_note.rst

.. note::

   The project was recently renamed from its internal name of **Securing AI Testbed** to **Dioptra**, and updating all usages of the old name is a work-in-progress.

.. include:: overview/executive-summary.rst

.. TODO: Delete this line and uncomment once repository is made public
..
.. Getting Started
.. ---------------

.. The testbed is available on GitHub at https://github.com/usnistgov/dioptra.
.. Complete documentation, including a Quick Start guide, can be found here: [url stub]

Points of Contact
-----------------

Email us: ai-nccoe@nist.gov


.. toctree::
   :hidden:
   :maxdepth: -1

   glossary

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Getting Started

   getting-started/installation
   getting-started/newcomer-tips

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: User Guide

   user-guide/the-basics
   user-guide/entry-points
   user-guide/task-plugins
   user-guide/custom-entry-points
   user-guide/custom-task-plugins
   user-guide/generics-plugin-system
   user-guide/task-plugins-collection
   user-guide/api-reference-sdk
   user-guide/api-reference-restapi

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Tutorials and Examples

   tutorials/example-basic-mlflow-demo
   tutorials/example-tensorflow-mnist-classifier
   tutorials/example-tensorflow-mnist-feature-squeezing
   tutorials/example-tensorflow-mnist-model-inversion
   tutorials/example-tensorflow-adversarial-patches
   tutorials/example-tensorflow-backdoor-poisoning
   tutorials/example-tensorflow-imagenet-resnet50-fgm
   tutorials/example-tensorflow-imagenet-pixel-threshold
   tutorials/example-pytorch-mnist-membership-inference

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Deployment Guide

   deployment-guide/system-requirements
   deployment-guide/docker-images-list-and-settings
   deployment-guide/single-machine-deployment
   deployment-guide/task-plugins-management
   deployment-guide/obtaining-datasets
   deployment-guide/testbed-ansible-collection

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Developer Guides

   dev-guide/index
   dev-guide/programming-style
   dev-guide/design-architecture
   dev-guide/design-restapi
   dev-guide/design-sdk
