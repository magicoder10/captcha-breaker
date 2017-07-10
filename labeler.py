import base64
import json
import cv2
import tkinter as tk

from io import BytesIO

from PIL import Image, ImageTk

from captcha.preprocessor import get_digits

with open('images.json') as file:
    images_json = file.read()

images = json.loads(images_json)['images']
examples = []

for i in range(len(images)):
    image_string = BytesIO(base64.b64decode(images[i]['jpg_base64']))
    image = Image.open(image_string)
    first_character, second_character, third_character, fourth_character = get_digits(image)
    if first_character is not None:
        captcha_filename = 'images/%d.png' % i
        first_filename = 'digits/%d_%d.png' % (i, 0)
        second_filename = 'digits/%d_%d.png' % (i, 1)
        third_filename = 'digits/%d_%d.png' % (i, 2)
        fourth_filename = 'digits/%d_%d.png' % (i, 3)

        example = {
            'captcha_image': captcha_filename,
            'digit_images': [
                first_filename,
                second_filename,
                third_filename,
                fourth_filename
            ]
        }
        examples.append(example)
        image.save(captcha_filename)
        cv2.imwrite(first_filename, first_character)
        cv2.imwrite(second_filename, second_character)
        cv2.imwrite(third_filename, third_character)
        cv2.imwrite(fourth_filename, fourth_character)

print('Conversion rate: %f' % (len(examples) / len(images)))

# if len(examples) > 300:
#     examples = examples[:300]

good_examples = []


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()
        self.curr_example_index = -1
        self.next()

    def create_widgets(self):
        self.captcha_canvas = tk.Canvas(self.master, width=100, height=50)
        self.captcha_canvas.pack()

        label = tk.Label(self.master, text="Who are they?")
        label.pack()
        self.entry = tk.Entry(self.master)
        self.entry.pack()
        self.entry.bind('<Return>', self.enter)

        self.next_button = tk.Button(self.master, text="Next", command=self.saveAndLoadNext)
        self.next_button.pack()

        self.save_button = tk.Button(self.master, text="Save", command=self.save)
        self.save_button.pack()

        self.skip_button = tk.Button(self.master, text="Skip", command=self.skip)
        self.skip_button.pack()

    def setCaptcha(self, image_filename):
        captcha = ImageTk.PhotoImage(file=image_filename)
        self.captcha_canvas.captcha = captcha
        self.captcha_canvas.create_image((captcha.width() / 2, captcha.height() / 2), image=captcha)

    def enter(self, event):
        self.saveAndLoadNext(event)

    def save(self):
        examples_json = json.dumps(good_examples)
        with open('examples/%d.examples.json' % len(good_examples), 'w') as examples_json_file:
            examples_json_file.write(examples_json)
        print('Saved %d examples to examples/%d.examples.json' % (len(good_examples), len(good_examples)))

    def skip(self):
        print('Skip for %s: => %d/%d => %d' % (examples[self.curr_example_index]['captcha_image'],
                                                   self.curr_example_index + 1, len(examples),
                                                   len(good_examples)
                                                   ))

        self.next()

    def next(self):
        self.curr_example_index += 1
        self.setCaptcha(examples[self.curr_example_index]['captcha_image'])
        self.entry.delete(0, tk.END)

    def saveAndLoadNext(self):
        label = self.entry.get()
        examples[self.curr_example_index]['label'] = label
        good_examples.append(examples[self.curr_example_index])
        print('Label for %s: %s => %d/%d => %d' % (examples[self.curr_example_index]['captcha_image'],
                                                   label,
                                                   self.curr_example_index + 1, len(examples),
                                                   len(good_examples)
                                                   ))
        if self.curr_example_index < len(examples) - 1:
            self.next()
        else:
            self.save()
            self.master.destroy()


root = tk.Tk()
root.title("Captcha labeler")
root.geometry("200x200")
app = Application(master=root)
root.lift()
app.mainloop()
