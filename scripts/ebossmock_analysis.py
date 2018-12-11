"""Simple script to run analysis on eBOSS mocks"""

from pkg_resources import resource_filename
from dla_cnn.data_model.eboss_mocks import process_catalog_eboss_mock


def main():
    spec_path = '/home/xavier/DESI/eBOSS/'
    spec_file = spec_path + 'spec-n1.2.fits'
    cat_file = spec_path + 'catalog_10000.fits'

    default_model = resource_filename('dla_cnn', "models/model_gensample_v7.1")

    process_catalog_eboss_mock(kernel_size=400, model_checkpoint=default_model,
                               spec_file=spec_file, cat_file=cat_file,
                               output_dir="./", make_pdf=False, debug=False, num_cores=12)

# Command line execution
if __name__ == '__main__':
    main()
