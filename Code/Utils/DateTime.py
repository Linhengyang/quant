from datetime import datetime
from datetime import timedelta




# get series of dates between {startdate:YYYYmmdd} and {enddate:YYYYmmdd} in every {gapday:int >= 1} days.
# startdate would be the first day.
# return type: list of strings of YYYYmmdd format dates

def yld_natural_dates(
        startdate:str,
        enddate:str,
        gapday:int,
        dateformat="%Y%m%d"
        ):
    '''
    A generator, yielding natural date from startdate to enddate with gap gapday
    '''
    startdt = datetime.strptime(startdate, dateformat)

    period_days = (datetime.strptime(enddate, dateformat) - \
                   datetime.strptime(startdate, dateformat)
                   ).days
    
    for i in range(0, period_days+1, gapday):

        yield (startdt + timedelta(days=i)).strftime(dateformat)



    
    
















if __name__ == "__main__":
    datelist = list( yld_natural_dates('20231220', '20240104', 2) )
    print(datelist)