# Data-Science-Jobs-Salaries

work_year The year during which the salary was paid. There are two types of work year values: 2020 : Year with a definitive amount from the past 2021e : Year with an estimated amount (e.g. current year)

experience_level The experience level in the job during the year with the following possible values: EN : Entry-level / Junior MI : Mid-level / Intermediate SE : Senior-level / Expert EX : Executive-level / Director

employment_type The type of employement for the role: PT : Part-time FT : Full-time CT : Contract FL : Freelance

job_title The role worked in during the year.

salary The total gross salary amount paid.

salary_currency The currency of the salary paid as an ISO 4217 currency code.

salaryinusd The salary in USD (FX rate divided by avg. USD rate for the respective year via fxdata.foorilla.com).

employee_residence Employee's primary country of residence in during the work year as an ISO 3166 country code.

remote_ratio The overall amount of work done remotely, possible values are as follows: 0 : No remote work (less than 20%) 50 : Partially remote 100 : Fully remote (more than 80%)

company_location The country of the employer's main office or contracting branch as an ISO 3166 country code.

company_size The average number of people that worked for the company during the year: S : less than 50 employees (small) M : 50 to 250 employees (medium) L : more than 250 employees (large)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

df = pd.read_csv("/content/drive/MyDrive/Data Science Jobs Salaries.csv")
df.head()

df.isnull().sum()

df.info()

df.describe()

df['work_year']

# Cleaning work_year
df['work_year'] = np.where(df['work_year']=='2021e', '2021',df['work_year'])
df['work_year'][:5]

Let's replace the category abbreviation by mapping them with their respective names.

exp_lvl_map = {'EN': 'Entry level', 'EX': 'Executive level', 'MI': 'Mid level', 'SE': 'Senior level'}
emp_type_map = {'FT': 'Full-time', 'PT': 'Part-time', 'CT': 'Contract', 'FL': 'Freelance'}
emp_res_map = {
    'DE': 'Germany',
    'GR': 'Greece',
    'RU': 'Russia',
    'US': 'US',
    'FR': 'France',
    'AT': 'Austria',
    'CA': 'Canada',
    'UA': 'Ukraine',
    'NG': 'Nigeria',
    'PK': 'Pakistan',
    'IN': 'India',
    'GB': 'UK',
    'ES': 'Spain',
    'IT': 'Italy',
    'PL': 'Poland',
    'BG': 'Bulgaria',
    'PH': 'Philippines',
     'PT': 'Portugal',
    'HU': 'Hungary',
    'SG': 'Singapore',
    'BR': 'Brazil',
    'MX': 'Mexico',
    'TR': 'Turkey',
    'NL': 'Netherlands',
    'AE': 'UAE',
    'JP': 'Japan',
    'JE': 'Jersey',
    'PR': 'Puerto Rico',
    'RS': 'Serbia',
    'KE': 'Kenya',
    'CO': 'Colombia',
    'NZ': 'New Zealand',
    'VN': 'Vietnam',
    'IR': 'Iran',
    'RO': 'Romania',
    'CL': 'Chile',
    'BE': 'Belgium',
    'DK': 'Denmark',
    'CN': 'China',
    'HK': 'Hong Kong',
    'SI': 'Slovenia',
    'MD': 'Moldova',
     'LU': 'Luxembourg',
    'HR': 'Croatia',
    'MT': 'Malta'
}
comp_loc_map = {
    'DE': 'Germany',
    'GR': 'Greece',
    'RU': 'Russia',
    'US': 'US',
    'FR': 'France',
    'AT': 'Austria',
    'CA': 'Canada',
    'UA': 'Ukraine',
    'NG': 'Nigeria',
    'PK': 'Pakistan',
    'IN': 'India',
    'GB': 'UK',
    'ES': 'Spain',
    'IT': 'Italy',
    'PL': 'Poland',
    'PT': 'Portugal',
    'HU': 'Hungary',
      'SG': 'Singapore',
    'BR': 'Brazil',
    'MX': 'Mexico',
    'TR': 'Turkey',
    'NL': 'Netherlands',
    'AE': 'UAE',
    'JP': 'Japan',
    'KE': 'Kenya',
    'CO': 'Colombia',
    'NZ': 'New Zealand',
    'VN': 'Vietnam',
    'IR': 'Iran',
    'CL': 'Chile',
    'BE': 'Belgium',
    'DK': 'Denmark',
    'CN': 'China',
    'SI': 'Slovenia',
    'MD': 'Moldova',
    'LU': 'Luxembourg',
    'HR': 'Croatia',
    'MT': 'Malta',
    'CH': 'Switzerland',
    'AS': 'American Samoa',
    'IL': 'Israel'
}
comp_size_map={'L': 'Large', 'M': 'Medium', 'S': 'Small'}

df['experience_level'] = df['experience_level'].map(exp_lvl_map)
df['employment_type'] = df['employment_type'].map(emp_type_map)
df['employee_residence'] = df['employee_residence'].map(emp_res_map)
df['company_location'] = df['company_location'].map(comp_loc_map)
df['company_size'] = df['company_size'].map(comp_size_map)

# Converting 'remote_ratio' from 'int' to 'category'
df['remote_ratio'] = df['remote_ratio'].astype('category')

df['remote_ratio'] = np.where(df['remote_ratio'] == 0, 'No Remote', np.where(df['remote_ratio'] == 50, 'Part-time Remote', 'Full-time Remote'))

df.head()

Creating two separate dataframes for the years 2020 and 2021 for ease of analysis.

df_2021 = df[df['work_year'] == '2021'].drop(columns='work_year', axis=1)
df_2020 = df[df['work_year'] == '2020'].drop(columns='work_year', axis=1)

df["experience_level"].value_counts()

sns.countplot(x="experience_level",data=df)

We have seen that mid level(Intermediate) is the most one of them and the second level is the senior(Expert).

Average Salary by Experience Level

exp_salary = df.groupby('experience_level')['salary_in_usd'].mean()
plt.figure(figsize = (10,6))
ax = sns.barplot(x = exp_salary.index, y = exp_salary.values, palette = 'Reds')
plt.title('Average Salary by Experience Level', fontsize=12, fontweight='bold')
plt.xlabel('Experience Level', fontsize=12, fontweight='bold')
plt.ylabel('Average Salary (USD)', fontsize=12, fontweight='bold')

for container in ax.containers:
    ax.bar_label(container,
                padding = -50,
                fontsize = 17,
                bbox = {'boxstyle': 'circle', 'edgecolor': 'red', 'facecolor': 'yellow'},
                label_type="edge",
                fontweight = 'bold'


                )

# Customize the background color
ax.set_facecolor("#f4f4f4")

# Remove the grid lines
ax.grid(False)

plt.show()

Experienced professionals earn the highest average salary at approximately 194,931 USD. Seniors also receive a competitive average salary of about 153,062 USD. Mid-Level employees have an average salary of around 104,545 USD. Entry-Level positions offer a lower average salary, at approximately 78,546 USD.

Average Salary by Employment Type

#Group data by 'employment_type' and calculate the average salary for each type
emp_salary = df.groupby('employment_type')['salary_in_usd'].mean()

plt.figure(figsize = (10,6))
p = sns.barplot(y = emp_salary.values, x = emp_salary.index, palette = 'cool_r')
plt.title('Average Salary by Employment Type', fontsize=12, fontweight='bold')
plt.xlabel('Employment Type', fontsize=12, fontweight='bold')
plt.ylabel('Average Salary (USD)', fontsize=12, fontweight='bold')

for container in p.containers:
    plt.bar_label(container,
                padding = -50,
                fontsize = 17,
                bbox = {'boxstyle': 'rarrow', 'edgecolor': 'red', 'facecolor': 'yellow'},
                label_type="edge",
                fontweight = 'bold',
                rotation = 90

                 )

# Customize the background color
p.set_facecolor("#f4f4f4")

# Remove the grid lines
p.grid(False)

plt.show()

Full-Time employees have the highest average salary at approximately 138,298 USD. Contractors also earn a competitive average salary of about 113,447 USD. Freelancers and Part-Time workers have lower average salaries, at around 51,808 USD and 39,534 USD.

Average Salary by Job Title (Top 10)

# Group data by 'job_title' and calculate the average salary for each title
job_title_salary= df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending = False)

plt.figure(figsize = (10,6))
p = sns.barplot(x= job_title_salary.values[:10], y = job_title_salary.index[:10])

plt.title('Average Salary by Job Title (Top 10)', fontsize=12, fontweight='bold')
plt.xlabel('Average Salary (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Job Title', fontsize=12, fontweight='bold')

for container in p.containers:
    p.bar_label(container,

                bbox = {'boxstyle': 'circle', 'facecolor': 'yellow', 'edgecolor': 'red'},
                fontweight = 'bold'


               )
# Customize the background color
p.set_facecolor("#f4f4f4")

# Remove the grid lines
p.grid(False)
plt.show()

Data Science Tech Lead has the highest average salary at 375,000 USD. Cloud Data Architect and Data Lead also have notably high salaries. The top 10 job titles exhibit strong earning potential in the data science field.

Average Salary by Currency

# Group data by 'salary_currency' and calculate the average salary in USD for each currency
currency_salary  = df.groupby('salary_currency')['salary_in_usd'].mean()

plt.figure(figsize = (10,6))
p = sns.barplot(x = currency_salary.values , y  = currency_salary.index, palette = 'nipy_spectral')
plt.title('Average Salary by Currency (Converted to USD)', fontsize=12, fontweight='bold')
plt.xlabel('Average Salary (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Currency', fontsize=12, fontweight='bold')

for container in p.containers:
    p.bar_label(container,
                bbox = {'boxstyle': 'square', 'facecolor': 'white', 'edgecolor': 'red'},
                fontweight = 'bold'

               )
# Customize the background color
plt.show()

Employees receiving salaries in USD have the highest average salary at approximately 149,351 USD. Salaries in ILS (Israeli Shekel) are notably high, with an average of 423,834 USD. GBP (British Pound) and CHF (Swiss Franc) also offer competitive average salaries. Other currencies vary in average salaries, with AUD (Australian Dollar) and BRL (Brazilian Real) being among the lowest.

*Average Data Science Salaries by Location *

# Group the data by company_location and calculate the mean salary for each location
average_salaries_by_location = df.groupby('company_location')['salary_in_usd'].mean().reset_index()

# Sort the locations by average salary in descending order
average_salaries_by_location = average_salaries_by_location.sort_values(by='salary_in_usd', ascending=False)

# Select the top N locations to plot
top_n_locations = 10  # You can change this number as needed

# Create a bar chart to visualize average salaries by country
plt.figure(figsize=(12, 6))
p = sns.barplot(x='salary_in_usd', y='company_location', data=average_salaries_by_location.head(top_n_locations), palette = 'bright')
plt.title('Top {} Average Data Science Salaries by Location'.format(top_n_locations), fontsize=12, fontweight='bold' )
plt.xlabel('Average Salary (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Location', fontsize=12, fontweight='bold')

for container in p.containers:
    p.bar_label(container,
                fontsize = 12,
                bbox = {'boxstyle': 'larrow', 'edgecolor': 'red', 'facecolor': 'white'},
                label_type="edge",
                fontweight = 'bold'
               )

# Customize the background color
ax.set_facecolor("#f4f4f4")
plt.show()

In Illinois (IL), the average data science salary is notably high, at approximately 271,447 USD. Puerto Rico (PR) and the United States (US) also offer competitive average salaries, with approximately 167,500 USD and 151,801 USD, respectively. Russia (RU) and Canada (CA) have average data science salaries of around 140,333 USD and 131,918 USD, respectively. New Zealand (NZ), Bosnia and Herzegovina (BA), Ireland (IE), Japan (JP), and Sweden (SE) round out the top locations with varying average salaries.

Average Salary by Company Size

# Group data by 'company_size' and calculate the average salary for each size
company_size_salary = df.groupby('company_size')['salary_in_usd'].mean()

# Plot the average salary by company size
plt.figure(figsize=(10, 6))
p = sns.barplot(x=company_size_salary.index, y=company_size_salary.values, palette = 'rainbow')
plt.title('Average Salary by Company Size', fontsize=12, fontweight='bold')
plt.xlabel('Company Size',fontsize=12, fontweight='bold')
plt.ylabel('Average Salary (USD)', fontsize=12, fontweight='bold')

for container in p.containers:
    p.bar_label(container, fontweight = 'bold',
                padding = -12,
                fontsize=12,
                bbox = {'boxstyle': 'circle', 'facecolor': 'yellow', 'edgecolor': 'black'}

               )
plt.show()

Medium-sized companies offer the highest average salary at approximately 143,117 USD. Large companies follow with an average of about 118,306 USD. Small companies offer a lower average salary of around 78,227 USD.

Salary Distribution for Data Science Professionals

# Set a custom style
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(df['salary_in_usd'], kde=True, color='#c44e52', bins=20, ax=ax)

# Customize labels and title
ax.set_title('Salary Distribution for Data Science Professionals', fontsize=14, fontweight='bold')
ax.set_xlabel('Salary (USD)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')

# Remove y-axis ticks
ax.tick_params(axis='y', which='both', left=False)

# Customize the background color
ax.set_facecolor("#f4f4f4")

# Show the plot
plt.tight_layout()
plt.show()

The salary distribution for data science professionals is right-skewed, with a majority of professionals earning lower to mid-range salaries. A noticeable peak in the distribution suggests a concentration of professionals within a specific salary range. The KDE (Kernel Density Estimate) curve provides a smooth estimate of the distribution, showing a prominent peak.

Average Salary by Experience Level and Employment Type

# Calculate average salary for each combination of experience level and employment type
cost_effectiveness = df.groupby(['experience_level', 'employment_type'])['salary_in_usd'].mean().reset_index()

# Find the combination with the highest average salary (maximum cost-effectiveness)
best_combination = cost_effectiveness.loc[cost_effectiveness['salary_in_usd'].idxmax()]

# Create a bar plot using Seaborn with x and y axes swapped
plt.figure(figsize=(12, 8))

# Use a different color palette for a modern look
sns.set_palette(sns.color_palette('bright'))

ax = sns.barplot(x='experience_level', y='salary_in_usd', hue='employment_type', data=cost_effectiveness)
plt.xlabel('Experience Level', fontsize=14, fontweight='bold')
plt.ylabel('Average Salary (USD)', fontsize=14, fontweight='bold')
plt.title('Average Salary by Experience Level and Employment Type', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12, fontweight='bold')

for container in ax.containers:
    ax.bar_label(container, label_type="edge", color="black",
                 padding=6,
                 fontweight='bold',
                 fontsize=12,
                 bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': 'black'})

# Increase legend font size and make it bold
legend = plt.legend(title='Employment Type', fontsize=12, title_fontsize=14)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Set the background color to a light gray
ax.set_facecolor("#f4f4f4")

# Remove the grid lines
ax.grid(False)

# Show the plot
plt.show()

st Employment Type and Experience Level for Maximum Cost-Effectiveness:

Average Salary by Company Location and Company Size

cost_effectiveness = df.groupby(['company_location', 'company_size'])['salary_in_usd'].mean().reset_index().sort_values(by = 'salary_in_usd', ascending = False)[:20]

# Find the combination with the highest average salary (maximum cost-effectiveness)
best_combination = cost_effectiveness.loc[cost_effectiveness['salary_in_usd'].idxmax()]

# Create a bar plot using Seaborn with x and y axes swapped
plt.figure(figsize=(12, 8))

# Use a different color palette for a modern look
sns.set_palette(sns.color_palette('bright'))

ax = sns.barplot(x='company_location', y='salary_in_usd', hue='company_size', data=cost_effectiveness)
plt.xticks(rotation=90, fontsize=12, fontweight='bold')

lg = plt.legend(title='Company Size', title_fontsize=10, fontsize=10, loc='upper right')

# Set the background color to a light gray
ax.set_facecolor("#f4f4f4")

# Remove the grid lines
ax.grid(False)

# Show the plot
plt.show()

Insights In Illinois (IL), large companies tend to offer an average salary of $423,834 USD, meeting cost-effectiveness criteria.

Count Plot for Experience Level and Employment Type etc

fig, axis = plt.subplots(2, 2, figsize=(10, 6))
sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})

columns = ['experience_level', 'employment_type', 'salary_currency', 'company_size']
axis = axis.flatten()
for i, col in enumerate(columns):
    p = sns.countplot(data=df, x=df[col], ax=axis[i], palette= "rainbow_r")
    axis[i].set_xticklabels(axis[i].get_xticklabels(), rotation=40)
    axis[i].set_xlabel(col.capitalize(), fontsize=12, fontweight='bold')
    axis[i].set_ylabel('Count', fontsize=12, fontweight='bold')

# Remove any remaining empty subplots
for j in range(len(columns), len(axis)):
    fig.delaxes(axis[j])

plt.tight_layout()
plt.show()

Insights Most common experience level: "Senior" (2518 counts). Most common employment type: "Full-Time" (3724 counts). Majority of salaries in USD (3229 counts). Most prevalent company size: "Medium" (3157 counts).

Start coding or generate with AI.
Employment Type

df["employment_type"].value_counts()

sns.countplot(x = 'employment_type', data=df)

We have seen that the most type of employment is the (Full time) and the others is approximately equal.

Job Title

df["job_title"].value_counts()

top_10_job_titles = df["job_title"].value_counts().sort_values(ascending=False).head(10)
top_10_job_titles

We have seen that the datascientist is the top jobs that appear here.

Company_location

df["company_location"].unique()

company_location_top_10 = df["company_location"].value_counts().sort_values(ascending=False).head(10)
company_location_top_10

We have seen that the most one is in US.

Salary group by jobtitle

job_title_group_by_salary = df[['salary','job_title']]
job_title_group_by_salary.sort_values(by=["salary", "job_title"],ascending=False).head(10)

Remote_ratio

df["remote_ratio"].value_counts()

sns.countplot(x="remote_ratio", data=df)

We have seen that the fully remote is the most one.

Group by Employment Type and Salary

EmploymentType_group_by_salary = df[['salary','employment_type']]
EmploymentType_group_by_salary.sort_values(by=["salary", "employment_type"],ascending=False).head(10)

arr = np.array(df.groupby('experience_level').salary.median())

exp_lev = np.array(['EN','EX','MI','SE'])

fig, ax = plt.subplots()

ax.bar(exp_lev,arr)
ax.set_title("Mean salary by each experience level",fontdict={'size':16,'color':'red'})
ax.set_xlabel("Experience level",fontdict={'size':13,'color':'red'})
ax.set_ylabel("Salary",fontdict={'size':13,'color':'red'})
ax.grid()

fig.set_size_inches(8,8)
plt.show()

insights:

In above plot we can know what is the avearge salary is being paid to by expereince level of employees. We can see that EX (Executives) are having highest mean salary

Salary group by experience level

salary_group_by_experience = df[["salary",'experience_level']]
salary_group_by_experience.sort_values(by=["salary",'experience_level'], ascending=False).head(15)

Salary group by employment type and job title

cols = df.groupby(['job_title', 'employment_type']).agg({'salary':['mean','min','max']})
cols.columns = ['salary_mean','salary_min','salary_max']
cols = cols.reset_index()
cols.head(15)

Company Size

df['company_size'].value_counts()

sns.countplot(x="company_size",data=df)

Company size group by salary in usd.

company = df.groupby(['company_size']).agg({'salary':'mean'})
company

We need to handle outliers in salary

df.describe()

sns.boxplot(x="salary_in_usd", data=df)
sns.stripplot(x='salary_in_usd', data=df, color="#474646")

g = sns.displot(x=df['salary_in_usd'], data=df, kde=True)
plt.title('Distribution of salary in USD')
g.fig.set_figwidth(10)
g.fig.set_figheight(6)

insights:

Above distritbution plots show us how salaries in USD is dstributed from lowest to highest salaries. As you can see there is only a salary which as high as 600k while most of them is around 80-90k.

q1 = df["salary_in_usd"].quantile(0.25)
q3 = df["salary_in_usd"].quantile(0.75)
iqr = q3 - q1
min_wisk = q1 - 1.5 * iqr
max_wisk = q3 + 1.5 * iqr
clean = df[df["salary_in_usd"].between(min_wisk, max_wisk)]

sns.displot(x="salary_in_usd", data=clean)

clean["salary_in_usd"].describe()

company = df.groupby(['company_size']).agg({'salary_in_usd':'mean'})
company

Employee Type Group By Remote Ratio

sns.countplot(x="employment_type",hue="remote_ratio", data=df)

We have seen that most of employers that fulltime work is fully remote and the percentage of employers that no remote is approximately no or very low percentage.

df["employee_residence"].unique()

Data Exploration`

df_2020.describe(include="all").T

df_2021.describe(include="all").T

From 2020 to 2021, the average salary has increased along with a decrease in standard deviation indicating a strong growth in salary. Remote ratio has increased by 7% from 2020 to 2021 aong with a 2% decrease in standard deviation indicating a favourable atmosphere for remote jobs. This can be due to pandemic or due to cost cutting by companies.

Distribution of Salary for Data science professionals from 2020 to 2021

fig = plt.figure(figsize=(12, 5), constrained_layout=True,facecolor='#E4E5E9', dpi=100)
spec = fig.add_gridspec(ncols=6, nrows=2)

ax0 = fig.add_subplot(spec[0, :4])
h1=sns.histplot(x=df_2021['salary_in_usd'], bins=50, kde=True, color='#657FFB', **{'edgecolor':'black'})
plt.ylabel("")
plt.xlabel('2021',labelpad=9, fontsize=13)
plt.xlim(0,700000)
plt.axvline(x=100269.91, color='r', linewidth=2, label="Mean")

ax1 = fig.add_subplot(spec[1, :4])
h2=sns.histplot(x=df_2020['salary_in_usd'], bins=50, kde=True, color='#BAD2EF', **{'edgecolor':'black'})
plt.xlim(0,700000)
plt.ylabel("")
plt.xlabel('2020',labelpad=9, fontsize=13)
plt.axvline(x=98910.89, color='r', linewidth=2, label="Mean")

ax2 = fig.add_subplot(spec[:, 4])
b2=sns.boxplot(y=df_2021['salary_in_usd'], color='#657FFB')
plt.xlabel('2021',labelpad=5, **{'fontsize':13})
plt.ylim(0,700000)
plt.ylabel('')

ax3 = fig.add_subplot(spec[:, 5])
b1=sns.boxplot(y=df_2020['salary_in_usd'], color='#BAD2EF')
plt.xlabel('2020',labelpad=5, **{'fontsize':13})
plt.ylabel('')
plt.ylim(0,700000)

fig.suptitle('Distribution of Salaries (in USD) of DataScience professionals', fontsize=16, y=1.07)
plt.show()

Number of data science professionals bagging salaries upto 200000 USD has increased over the period from 2020 to 2021. The peak salary has increased by about 150000 USD over the period from 2020 to 2021.

Time for Some Visualization

df['job_title'].nunique()

df['salary_in_usd']

mean_by_location = df.groupby("company_location")["salary_in_usd"].mean().sort_values(ascending = False)
plt.figure(figsize = (8, 8))
sns.barplot(y = mean_by_location.index, x = mean_by_location)

salary_by_title = df.groupby("job_title")["salary_in_usd"].mean().sort_values(ascending=False)
plt.figure(figsize=(10,10))
sns.barplot(x = salary_by_title, y = salary_by_title.index)

plt.figure(figsize = (10, 10))
sns.boxplot(x = df["salary_in_usd"], y = df["job_title"], order=salary_by_title.index)

df

unique_title = df['job_title'].unique()
unique_title_count = df['job_title'].nunique()
print(unique_title_count)
print(unique_title)

job_sal = ['job_title' ,'salary_in_usd', 'company_location', 'work_year']
new_salary_data = df[job_sal]

data_scientist = df.loc[df['job_title'] == 'Data Scientist']
data_scientist_sal = data_scientist[job_sal]

salary_dat_sci = data_scientist_sal.groupby(['company_location', 'work_year'])['salary_in_usd'].sum().sort_values(ascending=False)

for titles in unique_title:
    salaries = new_salary_data.loc[new_salary_data['job_title'] == titles]
    salary_list_titles = salaries.sort_values(['salary_in_usd'], ascending=False)
    print(salary_list_titles)

From the sheer amount of information regarding about the job titles in the dataframe, we are able to provide clarity and found out that there are 41 unique job titles circulating in the dataframe. With this we will proceed by finding out the average salary with the following job titles.

plt.figure(figsize=(15, 32))

job_salaries = df.groupby(['job_title'])['salary'].mean().sort_values(ascending=False)
title = [ tit for tit in job_salaries.index]
sns.barplot(x=job_salaries, y=title)
plt.xlabel('Average Salary in Millions')
plt.ylabel('Job Titles')

job_sal = ['job_title' ,'salary_in_usd', 'company_location', 'work_year']
new_salary_data = df[job_sal]

data_scientist = df.loc[df['job_title'] == 'Data Scientist']
data_scientist_sal = data_scientist[job_sal]

salary_dat_sci = data_scientist_sal.groupby(['company_location', 'work_year'])['salary_in_usd'].sum().sort_values(ascending=False)

for titles in unique_title:
    salaries = new_salary_data.loc[new_salary_data['job_title'] == titles]
    salary_list_titles = salaries.sort_values(['salary_in_usd'], ascending=False)
    print(salary_list_titles)

To provide us with a more meticulous layout, we have presented a sorted job titles with their following salary in USD. The columns of this dataframe are; job title, salary in USD, company location, work year. Simply, from the dataframe above, its what constitutes in the bargraph above, and derevied from its values the average of each salary in accordance with each job titles.

unique_location = df['company_location'].unique()
loc_count = df['company_location'].nunique()
print(unique_location)
print(f'The number of unique location is {loc_count}')

These are the unique company location. And, there are 41 of them.

These are the corresponding countries with thier dialing code:

DE - Germany US - United States RU - Russia FR - France AT - Austria CA - Canada UA - Ukraine NG - Nigeria IN - India ES - Spain PL - Poland GB - United Kingdom PT - Portugal DK - Denmark SG - Singapore MX - Mexico TR - Turkey NL - Netherlands AE - United Arab Emirates JP - Japan CN - China HU - Hungary KE - Kenya CO - Colombia NZ - New Zealand IR - Iran CL - Chile PK - Pakistan BE - Belgium GR - Greece SI - Slovenia BR - Brazil CH - Switzerland IT - Italy MD - Moldova LU - Luxembourg VN - Vietnam AS - American Samoa HR - Croatia IL - Israel MT - Malta

In the following cell, we will be, plotting a bargraph that represents the job titles existing in that certain company location and the following average salary in USD of the job title in that location.

#Job title
y= df['job_title'].value_counts()
x= df['job_title'].unique()
fig=plt.figure(figsize=(20,5))
plt.bar(x,y)
plt.xticks(rotation='vertical')
plt.show()

fig, ax = plt.subplots(41, 1, figsize=(10, 350));

for locations, axe in zip(unique_location, ax):
    company_loc = df['company_location'].isin([locations])
    new_sal_dat = new_salary_data[company_loc]
    salaries = new_sal_dat.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
    job_tit = [sal for sal in salaries.index]
    sns.barplot(x=salaries, y=job_tit, ax=axe)
    axe.set_title(f'Job posts with a high salary in - {locations}')
    axe.set_xlabel("Salary rate in USD")
    axe.set_ylabel("Job Titles")

Clearly, as presented in the graph, US usually has most of the job titles. It is obvious somehow that a productive country, it safe to assume that it offers a varied range of opputunity. Also the following countries that has more than or equal to five(5) job titles are; DE - Germany, US - United States, FR - France, CA - Canada, IN - India, ES - Spain, GB - United Kingdom.

employment = ['job_title', 'employment_type', 'salary_in_usd']
employ_data = df[employment]
unique_employ = df['employment_type'].unique()

for employ in unique_employ:
    employments = employ_data.loc[employ_data['employment_type'] == employ]
    employs = employments.sort_values(['salary_in_usd'], ascending = False)
    print(employs)

With the dataframe above, we can draw some insights. Out of 245 rows, 231 rows are constituted with an employment type of FT or full time employees. Out of 245 rows, 7 of it is constitued by PT or otherwise known as, part time. Also, 4 of out 245 rows are contituted by CT, or Contract. And, 3 our of 245 rows are contitued by FL, or Freelancers

fig, ax = plt.subplots(4, 1, figsize=(10, 50));

for emp, axe in zip(unique_employ, ax):
    emplo = df['employment_type'].isin([emp])
    new_employ_data = employ_data[emplo]
    grouped_employ = new_employ_data.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
    job_title2 = [jt for jt in grouped_employ.index]
    sns.barplot(x=grouped_employ, y=job_title2, ax=axe)
    axe.set_title(f'The average salary of job positions in employment type - {emp}')
    axe.set_xlabel("Salary rate in USD")
    axe.set_ylabel("Job Titles")

experience_level = ['job_title', 'experience_level', 'salary_in_usd']
experience_data = df[experience_level]
unique_exp_lvl = df['experience_level'].unique()

for xp in unique_exp_lvl:
    exper = experience_data.loc[experience_data['experience_level'] == xp]
    exper_sorted = exper.sort_values(['salary_in_usd'], ascending = False)
    print(exper_sorted)

Above is the dataframe with a sorted value of salary in descending order. Also, this dataframe pertains to the experience level of each job title.

fig, ax = plt.subplots(4, 1, figsize=(10, 50));

for exp, axe in zip(unique_exp_lvl, ax):
    ex = experience_data['experience_level'].isin([exp])
    new_exp_data = experience_data[ex]
    grouped_exp_data = new_exp_data.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
    job_title3 = [ title3 for title3 in grouped_exp_data.index ]
    sns.barplot(x=grouped_exp_data, y=job_title3, ax=axe)
    axe.set_title(f'The average salary of job positions in the experience level of - {exp}')
    axe.set_xlabel("Salary rate in USD")
    axe.set_ylabel("Job Titles")

Now, that we have visualized the different categories of experience level, namely; EN Entry-level, Junior MI Mid-level, Intermediate SE Senior-level, Expert EX Executive-level or Director.

The highest average salary from the category EN(Entry Level), is Machine Learning Scientist, having an average salary at about 250000 USD. And, the lowest salary, BI Data Analyst or Business Intelligence Data Analyst, having a salary of 9272 USD

In the category of SE(Intermediate, or Senior Level), the highest salary is approximately at 270000 USD with the job title of ML Engineer. the lowest job title salary is, Computer Vision Engineer, with a salary of about 47000 - 48000 USD.

In the EX(Expert or Executive-Level) catergory, the highest salary is Principal Data Engineer, with a salary of 600000 USD. The lowest salary in this category is about 65000 USD, and the job title is, Data Science Consultant.

Lastly, in the MI(Junior, or Mid-Level) category, the highest salary is about 480000 - 490000 USD, and the job title is Financial Data Analyst. The lowest in this category is 3D Computer Vision Researcher, with the salary of about 5000 USD

pd.options.display.float_format = '{:,.0f}'.format
country = df.groupby(['employee_residence', 'company_location','company_size','experience_level'])['salary_in_usd'].mean()
country = pd.DataFrame(data=country)
country.reset_index(inplace=True)
country.sort_values(by=['salary_in_usd'], inplace=True, ascending = False)
country.head(10)

countryplt = sns.barplot(data=country, x = 'employee_residence', y = 'salary_in_usd', palette = 'rocket')
countryplt.set_xticklabels(countryplt.get_xticklabels(), rotation=90, fontsize=8)
countryplt.set_title('Average Salaries by Country',fontsize=13);

plt.tick_params(axis='x', which='major', labelsize=8)

plt.tight_layout()

plt.figure(figsize=(10,5))

Next we will begin looking into which job titles in other countries have the highest salaries. We will use the Euro next because it is the next most common form of currency on the list.

fig = plt.figure(figsize=(14,12))
g = sns.barplot(x='job_title',
            y='salary',
            data=job_eur,
            ci=None)
g.bar_label(g.containers[0], rotation=45, label_type='center')
g.yaxis.get_major_formatter().set_scientific(False)
g.yaxis.get_major_formatter().set_useOffset(False)
plt.xlabel("Job Title", size=20)
plt.ylabel("Average Salary", size=20)
plt.title("Average Salary in EUR for Every Job Title", size=25)
plt.xticks(rotation=90);

Principle Data Scientists have the highest salary in european countries. The results are far different than the salaries of American data science careers. Machine Learning careers tend to be some of the highest paid in America, but in Europe that does not appear to be the case.

Simplify the data

We are now looking at a data set of full time data scientists and their salaries

usefull_data = df[df["employment_type"] == "FT"]
usefull_data.pop("employment_type")
usefull_data.describe()

Using Mutual Information relate features to Salary (USD)

Here, we explore which features best describe the salary of a data scientist

features = usefull_data.copy()
target = features.pop("salary_in_usd")

# Extract discrete features and do label encoding
discrete_features = features.select_dtypes("object").columns
for column in discrete_features:
    features[column],_ = features[column].factorize()

discrete_features_filt = features.columns.isin(discrete_features)
cont_features = features.columns[discrete_features_filt == False]

mi_scores_disc = mutual_info_classif(features[discrete_features], y = target)
mi_scores_disc = pd.Series(mi_scores_disc, index = discrete_features)
mi_scores_cont = mutual_info_regression(features[cont_features], y = target)
mi_scores_cont = pd.Series(mi_scores_cont, index = cont_features)

mi_scores = pd.concat([mi_scores_disc, mi_scores_cont]).sort_values(ascending = False)

sns.barplot(x = mi_scores, y = mi_scores.index)

Based on the mutual information scores, the data science salary has a strong dependence on location. This means that a data scientist can maximise their earning potential by relocating to a desired location.

However, for those who dont want to move, the salary depends strongly on experience and the job title.

Company location vs employee location

usefull_data["Lives_near_company"] = usefull_data["employee_residence"] == usefull_data["company_location"];
<ipython-input-39-4bcabb6e4e52>:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  usefull_data["Lives_near_company"] = usefull_data["employee_residence"] == usefull_data["company_location"];

usefull_data["Lives_near_company"].value_counts()

sns.swarmplot(x = usefull_data["Lives_near_company"], y = usefull_data["salary_in_usd"], hue = usefull_data["experience_level"])

So, most people live in the same place as the company. This is especially true for higher paid positions.

How much experience do you need for the different roles

# Declare indices and columns
exp = usefull_data["experience_level"].unique()
jobs = usefull_data["job_title"].unique()

# Make Data Frame
df = pd.DataFrame()
for i in jobs:
    for j in exp:
        df.loc[i, j] = 0

# Segregate jobs and experiences and count them
exp_vs_job = usefull_data.groupby("job_title")["experience_level"].value_counts()
for i in exp_vs_job.index.get_level_values(0):
    for j in exp_vs_job[i].index:
        df.loc[i, j] = exp_vs_job.loc[i, j]

df["tot"] = df.sum(axis=1)

# Sort the data frame
# df = df.sort_values(by = ["EN", "MI", "SE", "EX"], ascending = False)
df = df.apply(lambda x: x/df["tot"], axis = "rows")#.sort_values(by = ["EN", "MI", "SE", "EX"], ascending = False)

print("Entry Level: Blue \n Mid Level: Orange \n Senior Level: Green \n Executive Level: Red")
plt.figure(figsize = (14,10))
sns.barplot(y = df.index, x = 100*(df["EN"] + df["SE"] + df["MI"] + df["EX"]), color = "red", order=salary_by_title.index)
sns.barplot(y = df.index, x = 100*(df["EN"] + df["SE"] + df["MI"]), color = "green", order=salary_by_title.index)
sns.barplot(y = df.index, x = 100*(df["EN"] + df["MI"]), color = "orange", order=salary_by_title.index)
sns.barplot(y = df.index, x = 100*df["EN"], color = "blue", order=salary_by_title.index)

plt.xlabel("% of Data Scientists in each job type with a certain experience level");

This plot is a percentage barplot describing the proportion of people with these job roles which have a certain experience level. Each experience level will take up a proportionate amount of space. The actual size of the bars is irrelevant since we are looking at how much experience we need for each role.

Analysts and machine learning engineers can generally start at entry level and work their way up, however roles such as data science engineer or Big data architect require experience.

Experience vs Salary

sns.boxplot(x = usefull_data["experience_level"], y = usefull_data["salary_in_usd"], order = ["EN", "MI", "SE", "EX"])

Which job pays the best for each experience level

Entry Level

entry_level_data = usefull_data[usefull_data["experience_level"] == "EN"].sort_values(by="salary_in_usd", ascending=False)
plt.figure(figsize = (10, 5))
sns.boxplot(y = entry_level_data["job_title"], x = entry_level_data["salary_in_usd"])

Mid Level

entry_level_data = usefull_data[usefull_data["experience_level"] == "MI"].sort_values(by="salary_in_usd", ascending=False)
plt.figure(figsize = (10, 10))
sns.boxplot(y = entry_level_data["job_title"], x = entry_level_data["salary_in_usd"])

Senior Level

entry_level_data = usefull_data[usefull_data["experience_level"] == "SE"].sort_values(by="salary_in_usd", ascending=False)
plt.figure(figsize = (10, 10))
sns.boxplot(y = entry_level_data["job_title"], x = entry_level_data["salary_in_usd"])

Executive

entry_level_data = usefull_data[usefull_data["experience_level"] == "EX"].sort_values(by="salary_in_usd", ascending=False)
plt.figure(figsize = (10, 5))
sns.boxplot(y = entry_level_data["job_title"], x = entry_level_data["salary_in_usd"])

job_title Vs avg salary_in_usd

# Let's take a look at the jobs and their salaries in usd
sns.set()
job_salary_usd = df[['job_title', 'salary_in_usd']]
job_salary_usd = job_salary_usd.groupby('job_title').mean()['salary_in_usd'].sort_values().reset_index()
fig = plt.figure(figsize=(14,12))
g = sns.barplot(x='job_title',
                y='salary_in_usd',
                data=job_salary_usd,
                ci=None)
g.bar_label(g.containers[0], rotation=90)
plt.ylabel("Average Salary", size=20)
plt.xlabel('Job Title', size=20)
plt.xticks(rotation=90)
plt.title("Average Salary in USD for every Job Title", size=25);

It seems like financial analyst is earning more when the salary is in usd.

df['salary_currency'].nunique()

df['salary_currency'].unique()

Let's Focus on two of them. 'INR', 'JPY'.

job_jpy = df[df['salary_currency'] == 'JPY']
job_jpy

Well it seems like there is only one row related to japan😅😅. So, let's go with 'EUR'

job_titles and Avg salaries paid in INR

fig = plt.figure(figsize=(14,12))
g = sns.barplot(x='job_title',
            y='salary',
            data=job_inr,
            ci=None)
g.bar_label(g.containers[0], rotation=45, label_type='center')
g.yaxis.get_major_formatter().set_scientific(False)
g.yaxis.get_major_formatter().set_useOffset(False)
plt.xlabel("Job Title", size=20)
plt.ylabel("Average Salary", size=20)
plt.title("Average Salary in INR for Every Job Title", size=25)
plt.xticks(rotation=90);

It seems like Data Science manager is getting the highest pay in terms of INR, followed by Machine Learning Engineer. 3D Computer Vision Reasearchers are having the lowest average salary in INR

job_titles and Avg salaries paid in EUR

fig = plt.figure(figsize=(14,12))
g = sns.barplot(x='job_title',
                y='salary',
                data=job_eur,
                ci=None)
g.bar_label(g.containers[0],rotation=45,label_type='center')
plt.xlabel("Job Title", size=20)
plt.ylabel("Average Salary", size=20)
plt.xticks(rotation=90)
plt.title("Average Salary in EUR for every job title", size=25);

By the above graph the avg salary is highest for Principal Data Scientist when it comes to ERU. It is surprising to see that ML Engineers are having the lowest average salary in EUR

Top 5 Employee residence with highest salary paid

res = df.groupby('employee_residence').salary_in_usd.mean().sort_values(ascending=False)
res = res.head()
plt.figure(figsize=(10,5.5))
sns.barplot(x=res.index, y=res.values)
plt.title('highest salary by employee residence')
plt.ylabel('salary in USD')
plt.show()

Last 5 employee residence with least salary paid

res1 = df.groupby('employee_residence').salary_in_usd.mean().sort_values(ascending=True)
res1 = res1.head()

plt.figure(figsize=(10,10))
sns.barplot(x=res1.index, y=res1.values)
plt.title('lowest salary by employee residence')
plt.ylabel('salary in USD')
plt.show()

Relative share of employees by remote work

remo = np.array(df['remote_ratio'].value_counts(sort=True))

labels = ['Full remote', 'Partial remote', 'No remote']

plt.figure(figsize=(6,6))
plt.pie(remo, labels=labels, shadow=True, frame=True)
plt.legend()
plt.title('employees by remote ratio')
plt.show()

We can learn that more than 50% of the employees are working remote

Relative share of employees by company size

coms = df['company_size'].value_counts(sort=False)
print(coms)
Labels = [x for x in df['company_size'].unique()]
print('the company size labels are: ', Labels)
plt.figure(figsize=(6,6))
plt.pie(coms.values, labels=Labels, shadow=True, frame=True)
plt.legend()
plt.title('employees by company size')
plt.show()

We can learn that more than 50% of the employees are employed by large company and others by medium and small compnies

Pivot table of maximum salary by company size and remote ratio

t2 = pd.pivot_table(df,
                   values='salary_in_usd',
                   index='remote_ratio',
                   columns='company_size',
                   aggfunc= np.max
                   )
pd.DataFrame(t2)

Here we can see that at large and small comany 100% remote workers are getting paid higher than workers with 0% remote ration. it is possible because senior managers are wokring remote than juniors.

Number of employees by year of joining

plt.figure(figsize=(7,7))
sns.countplot(x=df['work_year'], data=df)
plt.title('Number of employees by year of joining')
plt.show()

insights:

almost 185 employees are estimated to join in 2021

Number of employees by Exp. level, Employment type, Company size and Remote ratio

columns = ['experience_level','employment_type','company_size']

ex =df['experience_level'].value_counts()

em =df['employment_type'].value_counts()

co =df['company_size'].value_counts()

re =df['remote_ratio'].map({100:'Full remote',
                           50:'Partial remote',
                           0:'No remote'}).value_counts()

# PLOT A FIGURE UISNG MATPLOTILB SUBPLOTS OBJECT

fig, axs = plt.subplots(2,2, figsize=(10,10))

axs[0,0].bar(ex.index,ex.values)
axs[0,0].set_xlabel('Experience level')
axs[0,0].set_ylabel('Number of employees')

axs[0,1].bar(em.index,em.values)
axs[0,1].set_xlabel('Employment type')
axs[0,1].set_ylabel('Number of employees')

axs[1,0].bar(co.index,co.values)
axs[1,0].set_xlabel('company size')
axs[1,0].set_ylabel('Number of employees')

axs[1,1].bar(re.index,re.values)
axs[1,1].set_xlabel('Remote ratio')
axs[1,1].set_ylabel('Number of employees')

plt.suptitle('Number of employees by Exp. level, Employment type, Company size and Remote ratio')
fig.tight_layout()
plt.show()

insights:

If you see by expereince level medium experienced employees are most hired in data science industry.

Majority of jobs are full time in data science industry.

Large companies hire more data scientists than medium and smaller ones.

As we have seen, majority of emplyees are wokring remotly.

#How much Companyies pay based on experience level
df.groupby(['company_size','experience_level'])['salary_in_usd'].mean().unstack().plot.bar()

#comapny employee distribtion by experience
df.groupby('company_size')['experience_level'].value_counts().unstack().plot.bar()

#distribution of workforce in the industry by experience
plotdata = df['experience_level'].value_counts().plot.pie(autopct='%1.1f%%')

#Year employees joined the domain
df.groupby('work_year')['work_year'].count().plot.pie(autopct='%1.1f%%')

#Distribution of employees based on employee residence
einc = df[df['employee_residence']== df['company_location']]
eninc = df[df['employee_residence']!= df['company_location']]
gloc = np.array([einc.count()['employee_residence'], eninc.count()['employee_residence']])

plt.pie(gloc,labels=['Employees Live in Company Location', 'Employees Live elsewhere'], autopct='%1.1f%%')

#How do employees work and their employment type
df.groupby(['employment_type','remote_ratio',])['employment_type'].count().unstack().plot.bar()

#How do companies hire and how are their employees expected to work?
df.groupby(['company_size','employment_type', 'remote_ratio'])['company_size'].count().unstack().plot.bar()

#salary by employment type
df.groupby('employment_type')['salary_in_usd'].mean().plot.bar()

#salary by experience level
df.groupby('experience_level')['salary_in_usd'].mean().plot.bar()

#salary by remote ratio
df.groupby('remote_ratio')['salary_in_usd'].mean().plot.bar()

#salary by company size
df.groupby('company_size')['salary_in_usd'].mean().plot.bar()

Which is the most popular job title?

from wordcloud import WordCloud,STOPWORDS

fig = plt.gcf()
fig.set_size_inches(15,8)
wc = WordCloud(stopwords=STOPWORDS,
              background_color='white',
              contour_width=3,
              contour_color='red',
              width=1250,
              height=800,
              max_words=250,
              max_font_size=250,
              random_state=42
              )

wc.generate(' '.join(df['job_title']))
fig= plt.imshow(wc, interpolation= "bilinear")
fig= plt.axis('off')

We can learn that Data scientist, Data engineer, Machine learning engieer are among the most popular titles

df.loc[:,['employee_residence','company_location']]

Distribution of employees by experience level and company size

def swarm_plot(x, y, hue, data=df):
    plt.figure(figsize=(6,6))
    sns.swarmplot(x=x, y=y, hue=hue,data=df)
    plt.show

plt.figure(figsize=(8,8))
sns.swarmplot(x=df['experience_level'],
              y=df['salary_in_usd'],
              hue=df['company_size'],
              data=df)
plt.title("Distribution of employees by experience level and company size")
plt.show

Distribution of employees by salaries

def strip_plot(x, y, hue, data=df):
    plt.figure(figsize=(6,6))
    sns.stripplot(x=x, y=y, hue=hue,data=df)
    plt.show

strip_plot(df['experience_level'],df['salary_in_usd'],df['company_size'], data=df)
strip_plot(df['employment_type'],df['salary_in_usd'],df['company_size'],data=df)
strip_plot(df['experience_level'],df['salary_in_usd'],df['remote_ratio'], data=df)
strip_plot(df['employment_type'],df['salary_in_usd'],df['remote_ratio'],data=df)

Do employees live in as same country as their company location??

lis = [df['company_location'].str.fullmatch(df['employee_residence'].at[idx], case=True).at[idx]
      for idx,i in enumerate(df['company_location'])]

print(lis)

pd.Series(lis).value_counts()

pd.Series(lis).map({True:"Same residence-company loc",
                    False:"Diff residence-company loc"}).value_counts().plot(kind='bar', rot=10,
                                                                            title='Number of employees living same or different country-company location',
                                                                            figsize=(10,7),
                                                                            xlabel='same/diff country company location',
                                                                            ylabel='Number of employees',
                                                                            fontsize=15,
                                                                              )

It is found that 209 out of 245 employees live in as same country as thier company.

Company Locations

com_loc = df.groupby('company_location').size().sort_values(ascending=False)
com_loc.head()

#plotting the data
fig = plt.figure(figsize=(14,12))
g = com_loc.plot(kind='bar')
g.bar_label(g.containers[0],rotation=45)
plt.xlabel("Location Count", size=20)
plt.ylabel('Location', size=20)
plt.title("Company Location Count", size=25)
plt.show()

Unironically USA is the location with most companies, the difference between other companies is vast.

Company Size

com_size = df.groupby('company_size').size().sort_values(ascending=False)
com_size.head()

fig = plt.figure(figsize=(14,12))
g = com_size.plot(kind='bar', color=['Red', 'pink', 'Green'])
g.bar_label(g.containers[0])
plt.title("Company Size Count", size=25)
plt.xlabel('Company Size', size=20)
plt.ylabel('Size Count', size=20)
plt.xticks(rotation=0)
plt.show()

Large Companies are high in numbers and Medium companies are lowest in number. I thought that smaller companies would be the lowest.

CONCLUSION:

In this dataset, we delved in the following information; location of company, experience level, employment type, and the average salary of each job titles.

Beginning with, there were 43 unique job titles in the dataset. The higest average salary out of 41 job titles is, Data Science Manager, with the average salary of about 2,700,000 - 2,800,000 USD.

In the location, there were 41 unique locations we have derived from the dataset; out of the 41 locations, US stood out, as it has a numerous range of jobs regarding data science. And the highest average salary is about 450,000 USD, and the job title is, Financial Data Analyst.

In the employment type, it is clear as it obviously is, being in a Full Time employment rewards most handsomely compared to any type of amployment. Highest salary in the FL(Full Time) employment still is, Financial Data Analyst, with the average salary of, 450,000 USD.

Lastly, in the experience level, still as obviously as it is, it is in the EX(Expert, Senior level) category that pays with the higest salary. And the highest is, Principal Data Engineer, with the average salary of, 600,000 USD.

📈 Salary trends:

Minimal change in 2020-2021. Continued fluctuations in 2021-2022. Similar pattern in 2022-2023. 🏢 Company size impact:

Medium companies see salary growth. Large companies have stable salaries. 👩‍🎓 Experience level:

Experienced pros earn most. Seniors follow, then mid-level, and entry-level. 💼 Employment types:

Full-Time has the highest average. Contractors also earn well. Freelancers and part-timers earn less. 🔝 Top job titles:

Data Science Tech Lead earns the most. 💱 Currency-based salaries:

USD salaries are highest. ILS, GBP, and CHF follow. 🌎 Top locations:

Illinois offers the highest salaries. 🏢 Company size impact:

Medium companies pay the most. 📊 Salary distribution:

Right-skewed with a peak. 🚀 Entry-level job titles:

Data Analyst, Data Scientist, Data Engineer. 💡 Cost-effective choice:

Experienced contractors earn 416,000 USD. 📊 Common insights:

"Senior" is the most common experience level. "Full-Time" is the most common employment type. Majority earn salaries in USD. "Medium" companies are prevalent.

Model Building For Salary Prediction

df1=df.drop(['experience_level','employment_type','job_title','salary_currency','employee_residence','remote_ratio','company_location','company_size'],axis=1)

X= df1.drop('salary', axis=1)
y= df1['salary'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Importing Machine learning Models
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

import pickle
** Linear Regression**

lm = LinearRegression()
lm.fit(X_train,y_train)

cross_val_score(lm,X_train,y_train, scoring='neg_mean_absolute_error')

cross_val_score(lm,X_train,y_train, scoring='neg_mean_absolute_error', cv=3)

# the above output is to skewed so by taking its mean we will be to read
np.mean(cross_val_score(lm,X_train,y_train, scoring='neg_mean_absolute_error', cv=2))

Lasso Regression

lm_l = Lasso()
np.mean(cross_val_score(lm_l, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))

It means that the lasso Regression model or Algorithm is worse than LinearRegresion according to our data

alpha = []
error = []

for i in range(1,1000):
    alpha.append(i/10)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train, y_train, scoring='neg_mean_absolute_error', cv=2)))

plt.plot(alpha,error)

Checking how much we improve our model

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns= ['Alpha', 'error'])

# checking how much we improve the model
df_err[df_err.error==max(df_err.error)]

Random Forest Model or Algorithm

rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))

Its quit awesome that Random Forest Model is doing for on our data.

without Minimizing the error it performing better than the last algorithms that we apply on our data.

** Tunning by GridsearchCV **

Lets understand how it works:

we will give the parameters that we want in model.
Based on those parameters it will analyse algorithm.
We will select which one is best according to these analysis of GridSearchCV.

parameters = {
    'n_estimators': range(10, 300, 10),
    'criterion': ('friedman_mse', 'absolute_error'),
    'max_features': (None, 'sqrt', 'log2')
}

gs = GridSearchCV(rf, parameters, error_score='raise', cv=3)
gs.fit(X_train, y_train)

gs.best_score_

gs.best_estimator_

we can achieve the accuracy of our model very high if we use the following parameters according to GridSearchCV method:

Parameters:

criterion should be absolute_error. max_features should be None. n_estimators should be 20.

Model Training or Test Ensembles:

Now we can train our because we know by now everything that are needed for our model to perform outstanding...

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)

mean_absolute_error(y_test,tpred_lml)

mean_absolute_error(y_test,tpred_rf)

mean_absolute_error(y_test,(tpred_lml+tpred_rf)/2)

This value should in between the value of tpred_lml and tpred_rf and it is in between so it correct and our model is not over tranied.

LinearRegression is suitable for our data which can be seen cause it has high value.

Putting the Model Into Production

Using pickle to store the neccessory values or variables into a file of pickle so that it can be used in the flask app....

import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump(pickl,open('model_file'+ '.p', "wb"))

file_name = "model_file.p"

with open (file_name, "rb") as pickled:
    data = pickle.load(pickled)
    model= data["model"]

model.predict(X_test.iloc[1,:].values.reshape(1,-1))
