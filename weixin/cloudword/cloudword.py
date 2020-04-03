# -*- coding: utf-8 -*-
import jieba.analyse
import codecs
import matplotlib.pyplot as plt
from scipy.misc import imread
from wordcloud import WordCloud,ImageColorGenerator
import sys
reload(sys)
import locale
sys.setdefaultencoding('utf8')

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    s = "hsl(0, 0%%, %d%%)" % 0
    return s


def generate_image():
    data = []
    jieba.analyse.set_stop_words("./stopwords.txt".encode())
    locale.setlocale(locale.LC_ALL, 'chs')

    with codecs.open("../news.txt", "r", encoding="utf8") as f:
        for text in f.readlines():
            #text = jieba.cut(text, cut_all= False)
            #data.extend(text)
            data.extend(jieba.analyse.extract_tags(text,topK= 200))


        data = " ".join(data)
        mask_img = imread('./ico1.jpg')

        wordcloud = WordCloud(
            font_path="msyh.ttc",  # 字体
            background_color="white",  # 背景颜色
            max_words=2000,  # 词云显示的最大词数
            mask=mask_img,  # 设置背景图片
            width=900,
            height=600,
            #scale=4.0,
            max_font_size=400,  # 字体最大值
            random_state=42,
        ).generate(data)
        image_colors = ImageColorGenerator(mask_img)
        #plt.imshow(wordcloud.recolor(color_func=grey_color_func), )
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.savefig('./weixin2.jpg',dpi = 1600)


if __name__ == '__main__':
    generate_image()



