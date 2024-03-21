from datetime import datetime
from datetime import timedelta




# get series of dates between {startdate:YYYYmmdd} and {enddate:YYYYmmdd} in every {gapday:int >= 1} days.
# startdate would be the first day.
# return type: list of strings of YYYYmmdd format dates

def yld_natural_dates(
        startdate: str,
        enddate: str,
        gapday: int,
        dateformat="%Y%m%d"
        ):
    '''
    A generator, yielding natural date from startdate to enddate(all can be included) with gap gapday
    '''
    startdt = datetime.strptime(startdate, dateformat)

    period_days = (datetime.strptime(enddate, dateformat) - \
                   datetime.strptime(startdate, dateformat)
                   ).days
    
    for i in range(0, period_days+1, gapday):

        yield (startdt + timedelta(days=i)).strftime(dateformat)



    
    



def first_day_of_week(
        date: str,
        dateformat="%Y%m%d"
        ) -> str:
    
    date = datetime.strptime(date, dateformat)
    day_ = date + timedelta(days = -date.weekday())

    return datetime.strftime(day_, dateformat)



def first_day_of_month(
        date: str,
        dateformat="%Y%m%d"
        ) -> str:
    
    date = datetime.strptime(date, dateformat)
    day_ = date.replace(day=1)

    return datetime.strftime(day_, dateformat)



def first_day_of_quarter(
        date: str,
        dateformat="%Y%m%d"
        ) -> str:
    
    date = datetime.strptime(date, dateformat)
    if date.month in (1,2,3):
        day_ = date.replace(month=1, day=1)
    elif date.month in (4,5,6):
        day_ = date.replace(month=4, day=1)
    elif date.month in (7,8,8):
        day_ = date.replace(month=7, day=1)
    elif date.month in (10,11,12):
        day_ = date.replace(month=10, day=1)

    return datetime.strftime(day_, dateformat)




def first_day_of_year(
        date: str,
        dateformat="%Y%m%d"
        ) -> str:
    
    date = datetime.strptime(date, dateformat)
    day_ = date.replace(month=1, day=1)

    return datetime.strftime(day_, dateformat)










if __name__ == "__main__":
    # datelist = list( yld_natural_dates('20231220', '20240104', 2) )
    # print(datelist)
   test = first_day_of_year('20231211')
   print(test)