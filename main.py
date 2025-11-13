from datetime import datetime
from typing import List, Literal, Optional

import torch
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.messages import ImageUrl
from pydantic_ai.models.outlines import OutlinesModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
IMAGE_PATH = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/examples/images/trader-joes-receipt.jpg"
PROMPT = """
You are an expert at extracting information from receipts.
Please extract the information from the receipt. Be as detailed as possible, do not miss any information.
"""


class Item(BaseModel):
    name: str
    quantity: Optional[int]
    price_per_unit: Optional[float]
    total_price: Optional[float]

class ReceiptSummary(BaseModel):
    store_name: str
    store_address: str
    store_number: Optional[int]
    items: List[Item]
    tax: Optional[float]
    total: Optional[float]
    date: Optional[str] = datetime
    payment_method: Literal["cash", "credit", "debit", "check", "other"]


def get_model() -> OutlinesModel:
    """Get the Pydantic Outlines model to use to extract information from the receipt."""
    tf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16
    )
    tf_processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        device_map="auto"
    )
    return OutlinesModel.from_transformers(tf_model, tf_processor)


if __name__ == "__main__":
    model = get_model()
    agent = Agent(model, output_type=ReceiptSummary)
    result = agent.run_sync(
        [PROMPT, ImageUrl(IMAGE_PATH)],
        model_settings=ModelSettings(extra_body={'max_new_tokens': 1000})
    )
    print(result.output)
    # store_name="Trader Joe's"
    # store_address='401 Bay Street, San Francisco, CA 94133'
    # store_number=0
    # items=[
    #   Item(name='BANANA EACH', quantity=7, price_per_unit=0.23, total_price=1.61),
    #   Item(name='BAREBELLS CHOCOLATE DOUG',quantity=1, price_per_unit=2.29, total_price=2.29),
    #   Item(name='BAREBELLS CREAMY CRISP', quantity=1, price_per_unit=2.29, total_price=2.29),
    #   Item(name='BAREBELLS CHOCOLATE DOUG', quantity=1, price_per_unit=2.29, total_price=2.29),
    #   Item(name='BAREBELLS CARAMEL CASHEW', quantity=2, price_per_unit=2.29, total_price=4.58),
    #   Item(name='BAREBELLS CREAMY CRISP', quantity=1, price_per_unit=2.29, total_price=2.29),
    #   Item(name='T SPINDRIFT ORANGE MANGO 8', quantity=1, price_per_unit=7.49, total_price=7.49),
    #   Item(name='T Bottle Deposit', quantity=8, price_per_unit=0.05, total_price=0.4),
    #   Item(name='MILK ORGANIC GALLON WHOL', quantity=1, price_per_unit=6.79, total_price=6.79),
    #   Item(name='CLASSIC GREEK SALAD', quantity=1, price_per_unit=3.49, total_price=3.49),
    #   Item(name='COBB SALAD', quantity=1, price_per_unit=5.99, total_price=5.99),
    #   Item(name='PEPPER BELL RED XL EACH', quantity=1, price_per_unit=1.29, total_price=1.29),
    #   Item(name='BAG FEE.', quantity=1, price_per_unit=0.25, total_price=0.25),
    #   Item(name='BAG FEE.', quantity=1, price_per_unit=0.25, total_price=0.25)]
    # tax=7.89
    # total=41.98
    # date='2023-04-01'
    # payment_method='credit'