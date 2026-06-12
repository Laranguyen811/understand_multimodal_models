from vit_prisma.utils.templates import *
import os

templates_list = []
##TODO: Write a function to collect all templates
def get_all_templates(directory):
    for filename in os.listdir(directory):
        if (filename.endswith('.py') 
            and filename != 'get_templates.py'):
            module_name = filename[:-3]  # Remove the .py extension
            module = __import__(f'vit_prisma.utils.templates.{module_name}', 
                                fromlist=[module_name])
            for attr in dir(module):
                if (attr.isupper() 
                    and attr.endswith('_TEMPLATE')
                    and not attr.startswith('_')):  # Assuming templates are defined in uppercase and end with '_TEMPLATES'
                    templates_list.append(getattr(module, attr))
    return templates_list
