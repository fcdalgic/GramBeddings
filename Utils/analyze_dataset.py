from tld import get_tld
from Utils.DataUtils import GetDataAndLabelsFromFiles

LABEL_PHISH = 1
LABEL_LEGIT = 0
dataset_name = 'ISCXURL2016'
train_file = '../data/' + dataset_name + '/train.csv'
val_file = '../data/' + dataset_name + '/test.csv'

print('Step-1 : Reading data from dataset')
train_samples, train_labels = GetDataAndLabelsFromFiles(train_file, convert_to_array=False)
val_samples, val_labels = GetDataAndLabelsFromFiles(val_file, convert_to_array=False)
total_samples = train_samples + val_samples
total_labels = train_labels + val_labels

indexes_phish = [idx for idx, element in enumerate(total_labels) if element == LABEL_PHISH]
samples_phish = [total_samples[i] for i in indexes_phish]

indexes_legit = [idx for idx, element in enumerate(total_labels) if element == LABEL_LEGIT]
samples_legit = [total_samples[i] for i in indexes_legit]

print('Step-2 : Extracting URL information')

def ParseDomainInfos(samples):
    domains = list()
    toplevels = list()
    suffixes = list()
    len_samples = len(samples)
    for i, sample in enumerate(samples):
        i += 1
        if i % int(len_samples / 100) == 0:
            print('Completed : ', float(i / len_samples))

        try:
            # res = get_tld('https://www.' + legit, as_object=True) use this only for PDRCNN because they don't have protocol info
            res = get_tld(sample, as_object=True)
        except:
            continue

        domains.append(res.domain)
        toplevels.append(res.tld)
        suffixes.append(res.suffix)

    return domains, toplevels, suffixes


legit_domains, legit_toplevels , legit_suffixes = ParseDomainInfos(samples_legit)
phish_domains, phish_toplevels , phish_suffixes = ParseDomainInfos(samples_phish)


print('Step-3: Calculating domain and top level domain scores')
legit_unique_domains = len(set(legit_domains))
legit_unique_topLevels = len(set(legit_toplevels))
legit_unique_suffixes = len(set(legit_suffixes))

phish_unique_domains = len(set(phish_domains))
phish_unique_topLevels = len(set(phish_toplevels))
phish_unique_suffixes = len(set(phish_suffixes))

samples_legit_len_avg = sum(map(len, samples_legit)) / len(samples_legit)
samples_phish_len_avg = sum(map(len, samples_phish)) / len(samples_phish)
samples_total_len_avg = sum(map(len, samples_legit + samples_phish)) / len(samples_legit + samples_phish)

indexes_histogram_bin_00_16 = [idx for idx, element in enumerate(samples_legit) if len(element) < 16]
indexes_histogram_bin_16_32 = [idx for idx, element in enumerate(samples_legit) if
                               len(element) >= 16 and len(element) < 32]
indexes_histogram_bin_32_48 = [idx for idx, element in enumerate(samples_legit) if
                               len(element) >= 32 and len(element) < 48]
indexes_histogram_bin_48_64 = [idx for idx, element in enumerate(samples_legit) if
                               len(element) >= 48 and len(element) < 64]
indexes_histogram_bin_64_80 = [idx for idx, element in enumerate(samples_legit) if
                               len(element) >= 64 and len(element) < 80]
indexes_histogram_bin_80_96 = [idx for idx, element in enumerate(samples_legit) if
                               len(element) >= 80 and len(element) < 96]
indexes_histogram_bin_96_112 = [idx for idx, element in enumerate(samples_legit) if
                                len(element) >= 96 and len(element) < 112]
indexes_histogram_bin_112_128 = [idx for idx, element in enumerate(samples_legit) if
                                 len(element) >= 112 and len(element) < 128]
indexes_histogram_bin_over_128 = [idx for idx, element in enumerate(samples_legit) if len(element) >= 128]

num_legit_len_histogram_bin_00_16 = len(indexes_histogram_bin_00_16)
num_legit_len_histogram_bin_16_32 = len(indexes_histogram_bin_16_32)
num_legit_len_histogram_bin_32_48 = len(indexes_histogram_bin_32_48)
num_legit_len_histogram_bin_48_64 = len(indexes_histogram_bin_48_64)
num_legit_len_histogram_bin_64_80 = len(indexes_histogram_bin_64_80)
num_legit_len_histogram_bin_80_96 = len(indexes_histogram_bin_80_96)
num_legit_len_histogram_bin_96_112 = len(indexes_histogram_bin_96_112)
num_legit_len_histogram_bin_112_128 = len(indexes_histogram_bin_112_128)
num_legit_len_histogram_bin_over_128 = len(indexes_histogram_bin_over_128)

indexes_histogram_bin_00_16 = [idx for idx, element in enumerate(samples_phish) if len(element) < 16]
indexes_histogram_bin_16_32 = [idx for idx, element in enumerate(samples_phish) if
                               len(element) >= 16 and len(element) < 32]
indexes_histogram_bin_32_48 = [idx for idx, element in enumerate(samples_phish) if
                               len(element) >= 32 and len(element) < 48]
indexes_histogram_bin_48_64 = [idx for idx, element in enumerate(samples_phish) if
                               len(element) >= 48 and len(element) < 64]
indexes_histogram_bin_64_80 = [idx for idx, element in enumerate(samples_phish) if
                               len(element) >= 64 and len(element) < 80]
indexes_histogram_bin_80_96 = [idx for idx, element in enumerate(samples_phish) if
                               len(element) >= 80 and len(element) < 96]
indexes_histogram_bin_96_112 = [idx for idx, element in enumerate(samples_phish) if
                                len(element) >= 96 and len(element) < 112]
indexes_histogram_bin_112_128 = [idx for idx, element in enumerate(samples_phish) if
                                 len(element) >= 112 and len(element) < 128]
indexes_histogram_bin_over_128 = [idx for idx, element in enumerate(samples_phish) if len(element) >= 128]

num_phish_len_histogram_bin_00_16 = len(indexes_histogram_bin_00_16)
num_phish_len_histogram_bin_16_32 = len(indexes_histogram_bin_16_32)
num_phish_len_histogram_bin_32_48 = len(indexes_histogram_bin_32_48)
num_phish_len_histogram_bin_48_64 = len(indexes_histogram_bin_48_64)
num_phish_len_histogram_bin_64_80 = len(indexes_histogram_bin_64_80)
num_phish_len_histogram_bin_80_96 = len(indexes_histogram_bin_80_96)
num_phish_len_histogram_bin_96_112 = len(indexes_histogram_bin_96_112)
num_phish_len_histogram_bin_112_128 = len(indexes_histogram_bin_112_128)
num_phish_len_histogram_bin_over_128 = len(indexes_histogram_bin_over_128)

print('Step-3: Calculating domain and top level domain scores')
header = 'dataset;'\
         'total samples;phish_samples;legit_samples;' \
         'legit_domains;legit_tlds;' \
         'phish_domains;phish_tlds;' \
         'samples_legit_len_avg;samples_phish_len_avg;samples_total_len_avg;' \
         'num_legit_len_histogram_bin_00_16;' \
         'num_legit_len_histogram_bin_16_32;' \
         'num_legit_len_histogram_bin_32_48;' \
         'num_legit_len_histogram_bin_48_64;' \
         'num_legit_len_histogram_bin_64_80;' \
         'num_legit_len_histogram_bin_80_96;' \
         'num_legit_len_histogram_bin_96_112;' \
         'num_legit_len_histogram_bin_112_128;' \
         'num_legit_len_histogram_bin_over_128;' \
         'num_phish_len_histogram_bin_00_16;' \
         'num_phish_len_histogram_bin_16_32;' \
         'num_phish_len_histogram_bin_32_48;' \
         'num_phish_len_histogram_bin_48_64;' \
         'num_phish_len_histogram_bin_64_80;' \
         'num_phish_len_histogram_bin_80_96;' \
         'num_phish_len_histogram_bin_96_112;' \
         'num_phish_len_histogram_bin_112_128;' \
         'num_phish_len_histogram_bin_over_128'

outputs = list()
outputs.append(dataset_name)
outputs.append(len(samples_legit + samples_phish))
outputs.append(len(samples_phish))
outputs.append(len(samples_legit))

outputs.append(legit_unique_domains)
outputs.append(legit_unique_topLevels)

outputs.append(phish_unique_domains)
outputs.append(phish_unique_topLevels)

outputs.append(samples_legit_len_avg)
outputs.append(samples_phish_len_avg)
outputs.append(samples_total_len_avg)

outputs.append(num_legit_len_histogram_bin_00_16    )
outputs.append(num_legit_len_histogram_bin_16_32    )
outputs.append(num_legit_len_histogram_bin_32_48    )
outputs.append(num_legit_len_histogram_bin_48_64    )
outputs.append(num_legit_len_histogram_bin_64_80    )
outputs.append(num_legit_len_histogram_bin_80_96    )
outputs.append(num_legit_len_histogram_bin_96_112   )
outputs.append(num_legit_len_histogram_bin_112_128  )
outputs.append(num_legit_len_histogram_bin_over_128 )

outputs.append(num_phish_len_histogram_bin_00_16    )
outputs.append(num_phish_len_histogram_bin_16_32    )
outputs.append(num_phish_len_histogram_bin_32_48    )
outputs.append(num_phish_len_histogram_bin_48_64    )
outputs.append(num_phish_len_histogram_bin_64_80    )
outputs.append(num_phish_len_histogram_bin_80_96    )
outputs.append(num_phish_len_histogram_bin_96_112   )
outputs.append(num_phish_len_histogram_bin_112_128  )
outputs.append(num_phish_len_histogram_bin_over_128 )

values_str = ";"
values_str = values_str.join([str(x) for x in outputs])

print(header)
print(values_str)






