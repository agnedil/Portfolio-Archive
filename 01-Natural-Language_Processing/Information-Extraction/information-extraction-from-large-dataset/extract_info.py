import os


def get_prodingr(fullstr, smallstr):                                # return required elements from file - product name
                                                                    # or multiple ingredient names - as list
    result = []
    mylength = len(smallstr)
    posnum = fullstr.find(smallstr)                                 # returns "-1" if no occurrences
    if posnum == -1:
        print("No product or ingredients!")

    while posnum != -1:
        start = posnum + mylength                                   # find the end of smallstr - start of name
        end = fullstr.find('\n', start)                             # find the next eol from start - end of name
        foundName = fullstr[start:end].strip()
        foundName = foundName.replace("\t", " ")
        while ';' in foundName:                                     # replace semicolns as they are used to separate ultimate categories
            foundName = foundName.replace(";", ",")
        while "  " in foundName:                                    # remove double spaces
            foundName = foundName.replace("  ", " ")
        result.append(foundName)
        posnum = fullstr.find(smallstr, end+1)

    return result


def get_reactivity(fileToStr):

    noReactivity = False                                            # is there no reactivity section in this MSDS?
    findme1 = 'Stability Indicator/Materials to Avoid:'             # titles of various sections as reference points
    findme2 = 'Stability Condition to Avoid:'
    findme3 = 'Hazardous Decomposition Products:'
    findme4 = 'Conditions to Avoid Polymerization:'
    findme5 = 'Stability Condition to\n    Avoid:'

    if findme5 in fileToStr:                                        # remove this weird feature I noticed
        fileToStr = fileToStr.replace(findme5, findme2)

    if findme1 in fileToStr:                                        # start of "Stability Indicator/Materials to Avoid:"
        start = fileToStr.find(findme1) + len(findme1)
    elif 'Reactivity' in fileToStr:
        temp = fileToStr.find('Reactivity')
        start = fileToStr.find('\n', temp) +1
    else:
        start = len(fileToStr)
        noReactivity = True                                         # no reactivity section in this MSDS

    if findme2 in fileToStr:                                        # end of "Stability Indicator/Materials to Avoid:"
        end = fileToStr.find(findme2, start)                        # or "Reactivity" section in the MSDS
    elif findme3 in fileToStr:
        end = fileToStr.find(findme3, start)
    elif findme4 in fileToStr:
        end = fileToStr.find(findme4, start)
    else:
        end = fileToStr.find("=", start)

    if noReactivity:
        reactivity = " "                                            # no reactivity section in this MSDS
    else:
        reactivity = fileToStr[start:end - 1]                       # the is the reactivity section

    stopwords = ['NONE', 'INDICATED', 'YES', 'SPECIFIED', 'BY', 'MANUFACTURER', 'KNOWN',
                 'RECOGNIZED', 'SEE', 'SUPP', 'DATA', 'NONE TO MANUFACTURERS KNOWLEDGE',
                 'REASONABLY', 'FORESEEABLE', 'REPORTED', 'NO', 'AVAILABLE']

    for word in stopwords:                                          # delete meaningless stopwords
        if word in reactivity:
            reactivity = reactivity.replace(word, "")

    while '\n' in reactivity:                                       # remove end of line
        reactivity = reactivity.replace('\n', ' ')

    while '.' in reactivity:                                        # remove periods
        reactivity = reactivity.replace('.', ' ')

    if 'E G ' in reactivity:                                        # restore the periods I need
        reactivity = reactivity.replace('E G ', 'E.G.')

    while ';' in reactivity:                                        # remove semicolon used for category separation
        reactivity = reactivity.replace(';', ',')

    while '  ' in reactivity:                                       # remove double spaces between words
        reactivity = reactivity.replace('  ', ' ')

    reactivity = reactivity.strip()                                 # remove head and training white spaces

    if len(reactivity) > 1000:                                      # get rid of annoying file that slips through
        reactivity = ""                                             # in whole no matter what

    return reactivity

def get_conditions_toAvoid(fileToStr):

    noConditions = False                                            # are there no conditions section in this MSDS?
    findme1 = 'Stability Condition to Avoid:'                       # titles of various sections as reference points
    findme2 = 'Hazardous Decomposition Products:'
    findme3 = 'Conditions to Avoid Polymerization:'
    findme4 = 'Stability Condition to\n    Avoid:'

    if findme4 in fileToStr:                                        # remove this weird feature I noticed
        fileToStr = fileToStr.replace(findme4, findme1)

    if findme1 in fileToStr:                                        # start of "Stability Condition to Avoid:"
        start = fileToStr.find(findme1) + len(findme1)
    else:
        start = len(fileToStr)
        noConditions = True                                         # no conditions in this MSDS

    if findme2 in fileToStr:                                        # end of "Stability Condition to Avoid:"
        end = fileToStr.find(findme2, start)                        # or "Reactivity" section in the MSDS
    elif findme3 in fileToStr:
        end = fileToStr.find(findme3, start)
    else:
        end = fileToStr.find("=", start)

    if noConditions:
        conditions = " "                                            # no conditions in this MSDS
    else:
        conditions = fileToStr[start:end - 1]

    stopwords = ['NONE', 'INDICATED', 'YES', 'SPECIFIED', 'BY', 'MANUFACTURER', 'KNOWN',
                 'RECOGNIZED', 'SEE', 'SUPP', 'DATA', 'NONE TO MANUFACTURERS KNOWLEDGE',
                 'REASONABLY', 'FORESEEABLE', 'REPORTED', 'NO', 'AVAILABLE']

    for word in stopwords:                                          # delete meaningless stopwords
        if word in conditions:
            conditions = conditions.replace(word, "")

    while '\n' in conditions:                                       # remove end of line
        conditions = conditions.replace('\n', ' ')

    while '.' in conditions:                                        # remove periods
        conditions = conditions.replace('.', ' ')

    if 'E G ' in conditions:                                        # restore the periods I need
        conditions = conditions.replace('E G ', 'E.G.')

    while ';' in conditions:                                       # remove semicolon used for category separation
        conditions = conditions.replace(';', ',')

    while '  ' in conditions:                                       # remove double spaces between words
        conditions = conditions.replace('  ', ' ')

    conditions = conditions.strip()                                 # remove head and training white spaces

    if len(conditions) > 1000:                                      # get rid of annoying file that slips through
        conditions = ""                                             # in whole no matter what

    return conditions

def get_ppe(fileToStr):

    findme = 'Exposure Controls/Personal Protection'
    ppe = ""
    if findme in fileToStr:                                 # if the section exists, otherwise return pp = ""
        temp = fileToStr.find(findme) + len(findme)
        start = fileToStr.find('\n', temp) + 1
        end = fileToStr.find('=', start) - 1
        ppe = fileToStr[start:end]
        while '\n' in ppe:                                  # remove eol for csv-style manipulations
            ppe = ppe.replace('\n', ' ')
        while ';' in ppe:                                   # remove semicolons, they are used to separate ultimate categories
            ppe = ppe.replace(';', ',')
        while '  ' in ppe:                                  # remove double spaces
            ppe = ppe.replace('  ', ' ')
        ppe = ppe.strip()
        if ppe.endswith('.'):                               # avoid periods before semicolon
            ppe = ppe[:-1]
    return ppe


result = open('results.txt', 'w+')
result.close()
mypath = '/home/andrew/Documents/2_UIUC/2_CS498_Cloud_Computing_Applications/Project/myMR/sample_data'
findme1 = 'Product ID:'
findme2 = 'Ingred Name:'

for root, dirs, files in os.walk(mypath):

    for file in files:
        with open(os.path.join(root, file)) as myfile:                  # fo through every file
            fileToStr = myfile.read()                                   # read file content into a string
        product = get_prodingr(fileToStr, findme1)                      # get product name
        ingredients = get_prodingr(fileToStr, findme2)                  # get ingredients
        if len(product) == 0:                                           # no product, skip this file
            print("No product")
            continue

        reactivity = get_reactivity(fileToStr)                          # get rectivity data

        conditions = get_conditions_toAvoid(fileToStr)                  # get conditions to avoid

        ppe = get_ppe(fileToStr)                                        # get ppe information

        with open('results.txt', 'a') as results:                       # write everything to csv file (sep = ';')
            results.write(product[0] + ';')

            if len(ingredients) == 0:
                results.write(';')
            else:
                for i in range(len(ingredients)):
                    if i == (len(ingredients) - 1):
                        results.write(ingredients[i] + "; ")
                        break;
                    else:
                        results.write(ingredients[i] + ", ")
            results.write(reactivity + ";")
            results.write(conditions + ";")
            results.write(ppe + ";\n")

        print(os.path.join(root, file) + " - finished")

print('All done!')