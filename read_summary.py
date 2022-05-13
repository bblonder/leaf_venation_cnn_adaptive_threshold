import pandas as pd
import os
import matplotlib.pyplot as plt

def make_graph(img_name):
    cwd = os.getcwd()
    img_prefix = img_name + "_img.png_cnn_vote_"
    folder_names = []
    for window_size in [32, 128, 512]:
        for vd in [0.2, 0.3]:
            if not (window_size == 32 and vd == 0.2):
                folder_names += [img_prefix + str(window_size) + "_" + str(vd)]
    for stat in ["MSTRatio", "Eccentricity", "Elongation", "Circularity"]:
        for f in folder_names:
            print(f)
            results = os.path.join(cwd, img_name, f, "results.xlsx")
            image_stats = os.path.join(cwd, img_name, f, f + ".xlsx")
            
            #results_sheet_num = 
            #results_wb = xlrd.open_workbook(results)
            #results_sheet = results_wb.sheet_by_index(sheet_num)

            image_stats_df = pd.read_excel(image_stats, sheet_name=6, engine='openpyxl')
            #getting mst ratio values
            # for rownum in range(1, image_stats_df.shape[0]):
            #     print(rownum)
            #     cval = image_stats_df.loc[str(rownum), str(mst_ratio_col)]
            #     mst_ratio.append(cval)

            #getting width_threshold values
            # for rownum in range(1, image_stats_df.shape[0]):
            #     cval = image_stats_df.loc[str(rownum), str(width_threshold_col)]
            #     width_threshold.append(cval)

            metric = image_stats_df[stat]
            width_threshold = image_stats_df["width_threshold"]


            # col_num = 10
            # arr_y = []
            # arr_x = []
            # for row_num in range(ref3d.rowxlo, row_lim):
            #     col_type = sheet.cell_type(row_num, col_num)
            #     if not col_type == xlrd.XL_CELL_EMPTY:
            #         cval = sheet.cell_value(row_num, col_num)

            plt.scatter(width_threshold, metric, label=f'{f}', s=4)
            plt.xlabel("Width Threshold")
            plt.ylabel(stat)
            plt.title(f"{stat} vs Width Threshold for {img_name}")
            plt.legend(prop={'size':6})
        plt.xscale('log')
        plt.savefig(os.path.join(cwd, img_name, f"{stat}_{img_name}.png"))
        plt.close()

