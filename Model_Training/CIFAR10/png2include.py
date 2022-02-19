import sys

from PIL import Image

if len(sys.argv) > 1:
    im = Image.open(sys.argv[1])
    print("input file is: ", sys.argv[1])
else:
    im = Image.open("test_input.png")
sqsize = min(im.size[1], im.size[0])
print("image ", im.size[0], " by ", im.size[1], " squaring to: ", sqsize)
im_crop = im.crop((0, 0, sqsize, sqsize))
size = 256, 256
print("and cropping to ", size)
im_crop.thumbnail(size, Image.ANTIALIAS)
print("and converting to greyscale")
imgbw = im_crop.convert("L")
imgbw.save('crop256.png')
idata = imgbw.getdata()
inum = 0
cnum = 0

with open('input_data.h', "w") as f:
    f.write('const int8_t input_int_bits = 2;\n')
    f.write('const int8_t input_dec_bits = 5;\n')
    f.write('const q7_t pIn[65536]__attribute__((section(".binSection"))) = {\n')
    for i in idata:
        inum = inum + 1
        cnum = cnum + 1
        if inum < 65536:
            print(i - 128, ",", file=f, end='')
        else:
            print(i - 128, file=f, end='')
        if cnum > 19:
            print("", file=f)
            cnum = 0
    f.write(" };")
    f.write("\n\n")
f.close()
