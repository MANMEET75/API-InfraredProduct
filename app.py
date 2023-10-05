import plotly.figure_factory as ff
import statsmodels.api as sm
import streamlit as st



st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(
    page_title="Infrared",
    page_icon="https://revoquant.com/assets/img/logo/logo-dark.png"
)




def main():

       


        st.sidebar.header("Statistical Methods for Numerical Variable")
        anomaly_options = ["None",
                           "Z-Score"
                
                           ]
        selected_anomalyAlgorithm = st.sidebar.selectbox("Select appropriate statistical techniques", anomaly_options)

        if selected_anomalyAlgorithm == "None":
            st.write(" ")
        elif selected_anomalyAlgorithm == "Z-Score":

            st.markdown("<h2 style='font-size: 24px; color: blue;'>Upload Dataset, Unleash Statistical Magic for Numerical Variables!</h2>",
                        unsafe_allow_html=True)
            data_file = st.file_uploader("Upload File", type=["csv", "xlsx", "XLSX"])

            if data_file is not None:
                file_extension = data_file.name.split(".")[-1]
                if file_extension == "csv":
                    data = pd.read_csv(data_file)
                elif file_extension in ["xlsx", "XLSX"]:
                    data = pd.read_excel(data_file)
                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")

                st.write(data.head())

                # Dealing with missing values
                threshold = 0.1
                missing_percentages = data.isnull().mean()
                columns_to_drop = missing_percentages[missing_percentages > threshold].index
                data = data.drop(columns=columns_to_drop)
                st.write(f"Features with more than {threshold * 100:.2f}% missing values dropped successfully.")

                # Dealing with duplicate values
                num_duplicates = data.duplicated().sum()
                data_unique = data.drop_duplicates()
                st.write(f"Number of duplicate rows: {num_duplicates}")

                # Display the cleaned data
                st.write("Cleaned Data:")
                st.write(data_unique.head(5))

                # Selecting numerical columns
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                newdf = data_unique.select_dtypes(include=numerics)

                # Select the column to use for z-score calculation
                column = st.selectbox("Select column for z-score", newdf.columns)

                # Set the threshold for anomaly detection
                threshold = st.slider('Threshold', 3, 10)

                # Calculate the z-score and perform anomaly detection
                data_with_anomalies_zscore = z_score_anomaly_detection(data_unique, column, threshold)


                # Determine colors for anomaly points and non-anomaly points
                data_with_anomalies_zscore['PointColor'] = 'Inlier'  # Default label for non-anomaly points
                data_with_anomalies_zscore.loc[data_with_anomalies_zscore['Anomaly'] == 1, 'PointColor'] = 'Outlier'  # Label for anomaly points
        
                # Create a scatter plot using Plotly
                fig = px.scatter(
                    data_with_anomalies_zscore,
                    x=column,
                    y="Anomaly",
                    color="PointColor",
                    color_discrete_map={"Inlier": "blue", "Outlier": "red"},
                    title='Z-Score Anomaly Detection',
                    labels={column: column, "Anomaly": 'Anomaly', "PointColor": "Data Type"},
                )

                # Update the trace styling
                fig.update_traces(
                    marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
                    selector=dict(mode='markers+text')
                )

                # Update layout with custom styling
                fig.update_layout(
                    legend=dict(
                        itemsizing='constant',
                        title_text='',
                        font=dict(family='Arial', size=12),
                        borderwidth=2
                    ),
                    xaxis=dict(
                        title_text=column,
                        title_font=dict(size=14),
                        showgrid=False,
                        showline=True,
                        linecolor='lightgray',
                        linewidth=2,
                        mirror=True
                    ),
                    yaxis=dict(
                        title_text='Anomaly',
                        title_font=dict(size=14),
                        showgrid=False,
                        showline=True,
                        linecolor='lightgray',
                        linewidth=2,
                        mirror=True
                    ),
                    title_font=dict(size=18, family='Arial'),
                    paper_bgcolor='#F1F6F5',
                    plot_bgcolor='white',
                    margin=dict(l=80, r=80, t=50, b=80),
                )

                # Display the Plotly figure using Streamlit's st.plotly_chart() function
                st.plotly_chart(fig)





                # Display the data with anomaly indicator
                st.write("Data with Anomalies:")
                st.write(data_with_anomalies_zscore)

                # Calculate the percentage of anomalies
                total_data_points = data_with_anomalies_zscore.shape[0]
                total_anomalies = data_with_anomalies_zscore["Anomaly"].sum()
                percentage_anomalies = (total_anomalies / total_data_points) * 100

                # Display the percentage of anomalies
                # Download button for the data with anomalies
                st.download_button(
                    label="Download Data",
                    data=data_with_anomalies_zscore.to_csv(index=False),
                    file_name="ZScoreAnomaly.csv",
                    mime="text/csv"
                )
                st.write(f"Percentage of Anomalies: {percentage_anomalies:.2f}%")





