import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(xml_files):
    xml_list = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train', 'valid', 'test']:
        image_path = os.path.join('artifacts/data_split', folder)
        image_files = glob.glob(image_path + '/*.jpg')
        xml_files = [os.path.join('artifacts/data/annotations/xmls', f.split(
            os.sep)[-1].replace('.jpg', '.xml')) for f in image_files]
        xml_df = xml_to_csv(xml_files)
        print(xml_df)
        outpath = os.path.join('artifacts/data_split',
                               folder, folder+'_labels.csv')
        print(outpath)
        xml_df.to_csv(outpath, index=None)
        print('Successfully converted xml to csv.')


# main()
