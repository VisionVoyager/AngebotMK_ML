from pdf2image import convert_from_path
import os
import tqdm

def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pdf2png(root_path, pdf_file_name):
    pdf_path = os.path.join(root_path, pdf_file_name)
    create_if_not_exists(pdf_path[:-4])
    print(pdf_path)
    pages = convert_from_path(pdf_path)
    for idx,page in tqdm.tqdm(enumerate(pages), total=len(pages)):
        png_path = os.path.join(pdf_path[:-4], f'{pdf_file_name[:-4]}_{idx}.png')
        page.save(png_path, 'PNG')

if __name__ == '__main__':
    market = '/home/tomislav/Projects/Angebot/LETOCI/VERO'
    for file in os.listdir(market):
        pdf2png(market, file)
    