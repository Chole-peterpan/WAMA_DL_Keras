import prettytable as pt
from function import char_color

tb = pt.PrettyTable()
tb.field_names = ["City name", "Area", "Population", "Annual Rainfall"]
tb.add_row(["Adelaide",1295, 1158259, 600.5])
tb.add_row(["Brisbane",5905, 1857594, 1146.4])
tb.add_row(["Darwin", 112, 120900, 1714.7])
tb.add_row(["Hobart", 1357, 205556,619.5])

tb.border = True
tb.junction_char=char_color('=',50,35)
tb.horizontal_char = char_color('=',50,32)

tb.vertical_char = char_color('|',50,33)

print(tb)