# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:38:50 2019

@author: foersterronny
"""

from reportlab.pdfgen import canvas

def SaveAt(settings):
    save_path = settings["Results"]["PDFPathAndFilename"]
    return save_path 
    

def WriteHello(settings):
    # create pdf
    c = canvas.Canvas(SaveAt(settings))
    c.drawString(100,750,settings)
    c.save()


def WriteOnNewLine(c, string, line_pos_old, indent, linewidth):
    line_pos_new = line_pos_old - linewidth
    c.drawString(indent,line_pos_new, string)

    return line_pos_new


def WriteLine(c, string, line_pos_old, indent):
    c.drawString(indent,line_pos_old, string)



    
def WriteSettings(settings):
    
    line_pos = 750
    linewidth = settings["Results"]["LineWidth"]
    indent = settings["Results"]["Indent"]
    indent_sub = settings["Results"]["Indent Sub"]
    
    #create pdf
    c = canvas.Canvas(SaveAt(settings))
    
    
    # write header
    line_pos = WriteOnNewLine(c, "Settings File", line_pos, linewidth)    
    

    for mykeys in settings.keys():
        line_pos = WriteOnNewLine(c, mykeys, line_pos, indent, linewidth)
        for mykeys_sub in settings[mykeys].keys():
            line_pos = WriteLine(c, mykeys, line_pos, indent_sub)
    
    c.save()