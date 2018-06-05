
from openpyxl import load_workbook

wb = load_workbook(filename='test2.xlsx')
sheet = wb['Sheet1']
# print(sheet['A1'].value)
# print(sheet)
# print('Sheet 1',sheet['1'])
# print(wb)

# for s, sh in enumerate(sheet['A']):
#     print(sh, sh.value)
#     if sh.value == 'حسن':
#         x = sh

# print(x.row, x.column)

# print(sheet['C'+str(x.row)].value)

location_list_rich = ['15', '64', '44', '70', '71', '69', '75', '76', '77', '104',
                      '105', '115', '17', '18', '19', '42', '43', '45', '55', '61',
                      '62', '63', '144', '169', '173', '174', '67', '81', '82']
print(len(location_list_rich))

location_list_poor = ['41', '46', '52', '83', '84', '85', '95', '97', '99', '101', '102',
                      '110', '111', '114', '116', '119', '120', '122', '123', '126', '129',
                      '132', '133', '135', '136', '138', '140', '150', '151', '154', '158',
                      '161', '165', '168', '180', '188', '22', '24', '25', '26', '28',
                      '29', '33', '48', '57', '65', '90', '92', '108', '121', '124', '128',
                      '134', '145', '148', '152', '153', '156', '157', '162', '164', '179',
                      '181', '182', '184', '185', '186', '187']
print(len(location_list_poor))
# input()

location_list_mid1 = ['10', '11', '12', '13', '16', '23', '30', '31', '39', '47',
                      '66', '91', '93', '94', '98', '100', '107', '109', '112', '117',
                      '130', '139', '141', '142', '160', '167', '172', '183', '14', '32',
                      '34', '36', '37', '38', '40', '49', '50', '51', '54', '56', '68', '78',
                      '79', '80', '96', '103', '106', '113', '118', '137', '143', '159',
                      '170', '171', '178',
                      '191', '195']
# print(len(location_list_mid1))
# input()

age_group = ['E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
# print(len(age_group))

sums_list = []
for c, col in enumerate(age_group):
    sum_1 = 0
    for l, ln in enumerate(location_list_mid1):
        sum_1 += int(sheet[col+ln].value)
    print('Population ', sheet[col+'2'].value, ':', sum_1)
    sums_list.append(sum_1)

population_total = 0
for p in sums_list:
    population_total += p

print('-'*50)
print('Population total mid', population_total)

print('-'*50)
temp = 0
for c, col in enumerate(age_group):
    num_samples = round((190 * sums_list[c]) / population_total)
    print('Num sample', sheet[col+'2'].value, ':', num_samples)
    temp += num_samples

print('-'*50)
print('temp', temp)


# Not need yet
def find_location_cell_in_sheet(column, location_name):
    for i, cell in enumerate(column):
        # if sh.value == location_name:
        #     return cell.row
        pass
