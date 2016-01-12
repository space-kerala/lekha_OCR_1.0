#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import Tkinter
import tkFileDialog
import PIL
from PIL import Image,ImageTk
import Pmw,sys
import preprocess as pp
import cv2
import codecs
import initial_temp as it
i=0

inp=open('compare_list.txt','r')
lines=inp.readlines()
class simpleapp_tk(Tkinter.Tk):
    def __init__(self,parent):
        Tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.initialize()

    def initialize(self):

        
        self.f=Tkinter.Frame(self, bg = "orange", width = 800, height = 500)
        self.f.pack(side=Tkinter.LEFT, expand=2, fill=Tkinter.BOTH)

        self.f1 = Tkinter.Frame(self.f, bg = "red", width = 500, height= 500)
        self.f1.grid(column=0,row=0,padx=5,pady=5, sticky='EW')

        self.f2 = Tkinter.Frame(self.f, bg = "red", width = 500, height=500)
        self.f2.grid(column=1,row=0,padx=5,pady=5)

        self.f3 = Tkinter.Frame(self.f, bg = "red", width = 500, height=500)
        self.f3.grid(column=2,row=0,padx=5,pady=5,sticky='EW')

        #for frame one

        self.source_url = Tkinter.StringVar()
        self.output_url=Tkinter.StringVar()
        self.current_url=Tkinter.StringVar()

        entry1 = Tkinter.Entry(self.f1,textvariable=self.source_url)
        entry1.grid(column=0,row=0,sticky='EW')
        # entry1.bind("<Return>",OnPressEnter)
        self.source_url.set(u'edited.png')

        # button = Tkinter.Button(self,text=u"Browse !",
        #                         command=self.OnButtonClick)
        browse_button = Tkinter.Button(self.f1,text=u"Browse !")
        browse_button.grid(column=1,row=0)
        
        s,t,u=self.get_req_info()
        self.source_url.set(s)
        self.output_url.set(t)
        self.current_url.set(u)
        print s,t,u,self.current_url.get()

        image=Image.open(s)
        image=image.resize((400,500), Image.ANTIALIAS)
        photo=ImageTk.PhotoImage(image)
        
        image_source=Tkinter.Label(self.f1,image=photo)
        image_source.image=photo
        image_source.grid(column=0,row=1,columnspan=2,sticky='EW')


        #for frame two

        self.folder_url = Tkinter.StringVar()
        entry2 = Tkinter.Entry(self.f2,textvariable=self.folder_url)
        entry2.grid(column=0,row=0,sticky='EW')
        # entry1.bind("<Return>",OnPressEnter)
        self.folder_url.set(u"Enter Folder file URL.")

        # button = Tkinter.Button(self,text=u"Browse !",
        #                         command=self.OnButtonClick)


        browse2_button = Tkinter.Button(self.f2,text=u"Browse !",command=self.OnButtonClick)
        browse2_button.grid(column=1,row=0)

        next_button=Tkinter.Button(self.f2,text="Next",command=self.onNext)

        prev_button=Tkinter.Button(self.f2,text="Previous",command=self.onPrevious)
        #       for the image
        image=Image.open(self.current_url.get())
        image=image.resize((100,100), Image.ANTIALIAS)
        photo=ImageTk.PhotoImage(image)
        
        image_frame=Tkinter.Label(self.f2,image=photo)
        image_frame.image=photo
        image_frame.grid(column=0,row=1,columnspan=2,sticky='EW')


        #for the forward and reverse buttons
        next_button.grid(column=1,row=2,pady=5,padx=5)

        prev_button.grid(column=0,row=2,pady=5,padx=5)

        #for the text section
        self.rec_char=Tkinter.StringVar()
        self.rec_entry=Tkinter.Entry(self.f2,textvariable=self.rec_char)
        self.rec_entry.bind("<Return>",self.onEnter)
        self.rec_entry.grid(column=0,row=3,columnspan=2)
        self.rec_char.set(lines[i][:-1])
        self.rec_entry.selection_range(0, Tkinter.END)
        #for save and edit buttons

        edit_button=Tkinter.Button(self.f2,text="Edit",command=self.onEdit)

        save_button=Tkinter.Button(self.f2,text="Save", command=self.onSave)

        edit_button.grid(column=0,row=4,pady=5,padx=5)

        save_button.grid(column=1,row=4,pady=5,padx=5)




        #for frame three
        
        filename=self.output_url.get()
        # filename.set(u"test.py")
        print filename
        text=Pmw.ScrolledText(self.f3,borderframe=5,vscrollmode='dynamic',hscrollmode='dynamic'
            ,labelpos='n',label_text='file %s' %filename,text_width=40, text_height=40,text_wrap='none')
        text.grid(row=0,column=0,sticky='EW')

        text.insert('end',open(str(filename),'r').read())
        #auxilary
        self.f.grid_columnconfigure(1,weight=1)
        # f2.grid_columnconfigure(0,weight=1)
        self.resizable(True,False)
        self.update()
        self.geometry(self.geometry())

    def OnButtonClick(self):
        print "you clicked"
        fname=tkFileDialog.askopenfilename()
        print fname
        self.labelVariable.set(fname)
        self.entryVariable.set(fname)
        # self.labelVariable.set( self.entryVariable.get()+" (You clicked the button)" )
        self.entry.focus_set()
        # myadditions
        self.display_image(str(fname))
        self.entry.selection_range(0, Tkinter.END)

    def get_req_info(self):
        # f=open('./compare_list.txt','w')
        # img=cv2.imread('./Example/dc_books_page.png',0)
        # if(img==None):
        #     print url+' does\'nt exist'
        #     exit()
        # img = pp.preprocess(img)
        # im,rot = pp.skew_correction(img)

        # line = pp.find_lines(im.copy())
        # # print len(linene)
        # label_list=train.label_unicode()
        # i=0
        # num=[]
        # for l in line:
        #     for w in l.word_list:
        #         for c in w.char_list:
        #             # num.append((str(i),label_list[int(c.label)]))
        #             tup=label_list[int(c.label)]
        #             f.write(tup+'\n')
        #             cv2.imwrite('samp/'+str(i)+'.png',c.data)
        #             i+=1
        # f.close()
        # self.source_url.set(u'./Example/dc_books_page.png')
        
        return './Example/dc_books_page.png','./output_file.txt','samp/0.png'
    def onNext(self):
        global i
        i+=1
        self.onChange()
        print i
        print self.current_url.get()
        self.current_url.set('samp/'+str(i)+'.png')
        image=Image.open(self.current_url.get())
        image=image.resize((100,100), Image.ANTIALIAS)
        photo=ImageTk.PhotoImage(image)
        
        image_frame=Tkinter.Label(self.f2,image=photo)
        image_frame.image=photo
        image_frame.grid(column=0,row=1,columnspan=2,sticky='EW')
        self.update()
        return 'samp/'+str(i)+'.png'

    def onPrevious(self):
        global i
        i=i-1
        self.onChange()
        print i
        if(i<0):
            i=0
        print self.current_url.get()
        self.current_url.set('samp/'+str(i)+'.png')
        image=Image.open(self.current_url.get())
        image=image.resize((100,100), Image.ANTIALIAS)
        photo=ImageTk.PhotoImage(image)
        
        image_frame=Tkinter.Label(self.f2,image=photo)
        image_frame.image=photo
        image_frame.grid(column=0,row=1,columnspan=2,sticky='EW')
        self.update()
        return 'samp/'+str(i)+'.png'
    def onChange(self):
        global i
        global lines
        print len(lines[i][:-1])
        self.rec_entry=Tkinter.Entry(self.f2,textvariable=self.rec_char)
        self.rec_entry.bind("<Return>",self.onEnter)
        self.rec_entry.grid(column=0,row=3,columnspan=2)
        if(lines[i][-1]=='\n'):
            lines[i]=lines[i][:-1]
        self.rec_char.set(lines[i])
        self.rec_entry.selection_range(0, Tkinter.END)
        self.update()
    def onEnter(self,event):
        global i,lines
        self.rec_char.set( self.rec_char.get() )
        print self.rec_char.get().encode('utf8')
        lines[i]=self.rec_char.get()        
    def onEdit(self):
        global i,lines
        self.rec_char.set( self.rec_char.get() )
        print self.rec_char.get().encode('utf8')
        lines[i]=self.rec_char.get().encode('utf8')
    def onSave(self):
        global lines
        global inp
        inp.close()
        inp=open('compare_list.txt','w')
        g=open('corrected.txt','w')
        for i in range(len(lines)):
            if(lines[i][-1]=='\n'):
                lines[i]=lines[i][:-1]
            g.write(lines[i])
            inp.write(lines[i]+'\n')
        g.close()
        inp.close()

        self.make_modified_file()
        filename='output_file.txt'
        # filename.set(u"test.py")
        print filename
        text=Pmw.ScrolledText(self.f3,borderframe=5,vscrollmode='dynamic',hscrollmode='dynamic'
            ,labelpos='n',label_text='file %s' %filename,text_width=50, text_height=50,text_wrap='none')
        text.grid(column=0,row=0,sticky='EW')

        text.insert('end',open(str(filename),'r').read())


# # for grid one
#         self.title1=Tkinter.Label(grid1,text=u'title1')
#         self.title1.grid(column=0,row=0,sticky='EW')

#         grid1.grid(column=0,row=0,sticky='EW')



# #for grid two
#         self.title2=Tkinter.Label(grid2,text=u'title2')
#         self.title1.grid(column=0,row=0,sticky='EW')
        
# #for grid three
    def make_modified_file(self):
        f=open('./compare_list.txt','r')
        g=open('./output_file.txt','w')
        img=cv2.imread('./Example/dc_books_page.png',0)
        if(img==None):
            print url+' does\'nt exist'
            exit()
        img = pp.preprocess(img)
        im,rot = pp.skew_correction(img)

        line = pp.find_lines(im.copy())
        # print len(linene)
        label_list=it.train.label_unicode()

        q=f.readlines()
        i=0
        num=[]
        for l in line:
            for w in l.word_list:
                for c in w.char_list:
                    # num.append((str(i),label_list[int(c.label)]))
                    tup=label_list[int(c.label)]
                    if(q[i][:-1]!=tup):
                        tup=q[i][:-1]
                    # f.write(tup+'\n')
                    g.write(tup)
                    # cv2.imwrite('samp/'+str(i)+'.png',c.data)
                    i+=1
                g.write(' ')
            g.write('\n')
        f.close()
        g.close()

if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('My App')
    app.mainloop()

