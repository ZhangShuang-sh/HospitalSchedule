# Hospital Shift Scheduling System

A Streamlit-based web application for generating fair and balanced monthly shift schedules for hospital departments.

## Features

- **Automatic Schedule Generation**: Creates monthly schedules with Day, Night, and 24-hour shifts
- **Fairness Optimization**: Balances weekend, night, and total shifts across staff (max difference of 1)
- **Flexible Staff Configuration**:
  - Set which staff can do night/24h shifts
  - Define fixed days off (supports ranges like `17-20`)
  - Pre-assign specific shifts (e.g., `1:Day,25:Night`)
- **Smart Constraints**:
  - Minimum 3-day gap between night shifts
  - Thursday night workers get weekend off
  - Max 1 shift per weekend per person
  - Holiday workers don't work weekends
- **Temporary Leave Handling**: Reschedule shifts when someone takes leave
- **Export**: Download schedules and statistics as CSV

## How to Use

### 1. Configure Staff

In the **Staff Setup** tab:
- Add staff members with their names and roles
- Check/uncheck **Night OK** and **24h OK** based on capabilities
- Enter **Fixed Off** dates (e.g., `5,10,15` or `17-20` for a range)
- Enter **Fixed On** for pre-assigned shifts (e.g., `1:Day` for Day shift on the 1st)

### 2. Set Schedule Parameters

In the sidebar:
- Select **Year** and **Month**
- Enter **Holiday dates** (comma-separated)
- Adjust shifts per day if needed

### 3. Generate Schedule

Click **Generate Schedule** to create the monthly schedule. The system will:
- Try 20 different combinations to find the fairest distribution
- Optimize shift balance through post-generation swaps
- Display the result with color-coded shifts

### 4. View Results

- **Schedule Tab**: View the monthly calendar with shifts highlighted
- **Statistics Tab**: See fairness metrics and individual shift counts
- **Export**: Download CSV files for further use

### 5. Handle Temporary Leave

If someone needs to take unexpected leave:
1. Go to the Schedule tab
2. Use the **Temporary Leave & Rescheduling** section
3. Select the person and enter leave dates
4. Click **Reschedule** to automatically reassign their shifts

## Shift Types

| Type | Symbol | Weight | Color |
|------|--------|--------|-------|
| Day | D | 1 | Green |
| Night | N | 1 | Blue |
| 24-Hour | 24 | 2 | Pink |

## Scheduling Rules

1. **Coverage**: Every day must have at least 1 Day shift and 1 Night shift
2. **Gap Rules**:
   - Night-to-Night: minimum 3 days (prefer 5)
   - Night-to-Day: minimum 3 days
   - Day-to-Day: minimum 3 days
3. **Weekend Rules**:
   - Max 1 shift per weekend per person
   - Thursday night = weekend off
   - Holiday workers don't work weekends
4. **Fairness**: Weekend and Night shifts balanced (max diff = 1)
5. **50% Target**: Staff with 14+ days off get 50% of normal shifts

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deployment

This app is designed for deployment on [Streamlit Community Cloud](https://streamlit.io/cloud).

1. Push the code to a GitHub repository
2. Connect your repository to Streamlit Community Cloud
3. Deploy with `app.py` as the main file

## Files

- `app.py` - Streamlit UI and main application
- `scheduler_logic.py` - Core scheduling algorithm
- `utils.py` - Helper functions and utilities
- `requirements.txt` - Python dependencies

## License

MIT License
