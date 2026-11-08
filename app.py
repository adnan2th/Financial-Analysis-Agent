
import streamlit as st
import os
import datetime
import autogen

from dotenv import load_dotenv
from autogen import ConversableAgent, AssistantAgent
from autogen.coding import LocalCommandLineCodeExecutor


st.set_page_config(page_title="AI Stock Analyzer", page_icon="📈")

st.title("📈 AI Stock Analyzer")
st.write("Groq + AutoGen + Python")
#load_dotenv()
# -------------------------
# Groq Configuration
# -------------------------
api_key = os.environ.get("groq_api")



llm_config = {
    "config_list": [
        {
            "model": "openai/gpt-oss-20b",
            "api_key": api_key,
            "base_url": "https://api.groq.com/openai/v1",
        }
    ]
}

# -------------------------
# Stock Functions
# -------------------------

def get_stock_prices(stock_symbols, start_date, end_date):
    """Download stock closing prices."""

    import yfinance

    stock_data = yfinance.download(
        stock_symbols,
        start=start_date,
        end=end_date
    )

    return stock_data.get("Close")


def plot_stock_prices(stock_prices, filename):
    """Plot stock prices."""

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    for column in stock_prices.columns:
        plt.plot(
            stock_prices.index,
            stock_prices[column],
            label=column
        )

    plt.title("Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# -------------------------
# User Input
# -------------------------

stock_symbols = st.text_input(
    "Enter stock symbols (comma separated)",
    "NVDA, TSLA"
)

if st.button("Analyze Stocks", type="primary"):

    with st.spinner("AI is analyzing stock prices..."):

        try:

            symbols = [
                symbol.strip().upper()
                for symbol in stock_symbols.split(",")
                if symbol.strip()
            ]

            if not symbols:
                st.error("Enter at least one stock symbol.")
                st.stop()

            today = datetime.date.today()
            start_date = today.replace(month=1, day=1)

            # -------------------------
            # Code Executor
            # -------------------------

            executor = LocalCommandLineCodeExecutor(
                timeout=60,
                work_dir="coding",
                functions=[
                    get_stock_prices,
                    plot_stock_prices
                ]
            )

            # -------------------------
            # Writer Agent
            # -------------------------

            code_writer_agent = AssistantAgent(
                name="code_writer_agent",
                llm_config=llm_config,
                code_execution_config=False,
                human_input_mode="NEVER",
            )

            system_message = (
                code_writer_agent.system_message
                + executor.format_functions_for_prompt()
            )

            code_writer_agent = ConversableAgent(
                name="code_writer_agent",
                system_message=system_message,
                llm_config=llm_config,
                code_execution_config=False,
                human_input_mode="NEVER",
            )

            # -------------------------
            # Executor Agent
            # -------------------------

            code_executor_agent = ConversableAgent(
                name="CodeExecutor",
                llm_config=False,
                code_execution_config={
                    "executor": executor
                },
                human_input_mode="NEVER",
                default_auto_reply=(
                    "Please continue. "
                    "If everything is done, reply 'TERMINATE'."
                ),
            )

            # -------------------------
            # Task
            # -------------------------

            message = f"""
            Today is {today}.

            Create a YTD stock price chart for:
            {symbols}

            Start date: {start_date}
            End date: {today}

            Use the provided functions:

            1. get_stock_prices()
            2. plot_stock_prices()

            Download the stock prices and plot them.
            Save the figure as ytd_stock_gains.png.

            Execute the Python code and finish with TERMINATE.
            """

            # -------------------------
            # Start Agents
            # -------------------------

            result = code_executor_agent.initiate_chat(
                code_writer_agent,
                message=message,
                max_turns=5
            )

            # -------------------------
            # Display Chart
            # -------------------------

            chart_path = os.path.join(
                "coding",
                "ytd_stock_gains.png"
            )

            if os.path.exists(chart_path):

                st.success("Stock analysis completed!")

                st.image(
                    chart_path,
                    caption="Year-to-Date Stock Prices"
                )

            else:
                st.warning("Chart was not generated.")

            with st.expander("View AI Conversation"):
                for msg in result.chat_history:
                    if msg.get("content"):
                        st.write(
                            f"**{msg.get('name', 'Agent')}:**"
                        )
                        st.write(msg["content"])

        except Exception as e:
            st.error(f"Error: {str(e)}")
