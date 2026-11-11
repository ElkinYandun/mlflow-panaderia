# clasificador-desperdicio @ challenger

- URI: `models:/clasificador-desperdicio@challenger`
- Versión: 1
- Endpoint: `POST http://127.0.0.1:5001/invocations`

## Esquema de entrada

```json
[
  {
    "type": "long",
    "name": "Customer_Age",
    "required": true
  },
  {
    "type": "string",
    "name": "Customer_Gender",
    "required": true
  },
  {
    "type": "string",
    "name": "Customer_Segment",
    "required": true
  },
  {
    "type": "string",
    "name": "Loyalty_Member",
    "required": true
  },
  {
    "type": "long",
    "name": "Year",
    "required": true
  },
  {
    "type": "long",
    "name": "Month",
    "required": true
  },
  {
    "type": "long",
    "name": "Day",
    "required": true
  },
  {
    "type": "string",
    "name": "Day_of_Week",
    "required": true
  },
  {
    "type": "string",
    "name": "Weekend",
    "required": true
  },
  {
    "type": "long",
    "name": "Hour",
    "required": true
  },
  {
    "type": "string",
    "name": "Season",
    "required": true
  },
  {
    "type": "string",
    "name": "Festival",
    "required": false
  },
  {
    "type": "string",
    "name": "Weather",
    "required": true
  },
  {
    "type": "double",
    "name": "Temperature",
    "required": true
  },
  {
    "type": "string",
    "name": "Product",
    "required": true
  },
  {
    "type": "string",
    "name": "Category",
    "required": true
  },
  {
    "type": "long",
    "name": "Quantity",
    "required": true
  },
  {
    "type": "double",
    "name": "Unit_Price",
    "required": true
  },
  {
    "type": "double",
    "name": "Cost_Price",
    "required": true
  },
  {
    "type": "double",
    "name": "Discount_Percentage",
    "required": true
  },
  {
    "type": "double",
    "name": "Selling_Price",
    "required": true
  },
  {
    "type": "double",
    "name": "Total_Bill",
    "required": true
  },
  {
    "type": "string",
    "name": "Payment_Method",
    "required": true
  },
  {
    "type": "string",
    "name": "Promotion_Applied",
    "required": true
  },
  {
    "type": "string",
    "name": "Promotion_Type",
    "required": false
  },
  {
    "type": "long",
    "name": "Shelf_Life_Days",
    "required": true
  },
  {
    "type": "string",
    "name": "Storage_Type",
    "required": true
  },
  {
    "type": "string",
    "name": "Perishability",
    "required": true
  },
  {
    "type": "string",
    "name": "Store_ID",
    "required": true
  }
]
```

## Esquema de salida

```json
[
  {
    "type": "tensor",
    "tensor-spec": {
      "dtype": "int64",
      "shape": [
        -1
      ]
    }
  }
]
```

## Petición

```bash
curl http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["Customer_Age", "Customer_Gender", "Customer_Segment", "Loyalty_Member", "Year", "Month", "Day", "Day_of_Week", "Weekend", "Hour", "Season", "Festival", "Weather", "Temperature", "Product", "Category", "Quantity", "Unit_Price", "Cost_Price", "Discount_Percentage", "Selling_Price", "Total_Bill", "Payment_Method", "Promotion_Applied", "Promotion_Type", "Shelf_Life_Days", "Storage_Type", "Perishability", "Store_ID"], "data": [[61, "Male", "Senior Citizen", "Yes", 2023, 9, 1, "Friday", "No", 15, "Monsoon", null, "Rainy", 26.6, "Soft Drink", "Beverages", 1, 44.47, 24.66, 27.8, 32.11, 32.11, "Wallet", "Yes", "Clearance", 257, "Dry Storage", "Low", "ST03"]]}}'
```

```json
{
  "dataframe_split": {
    "columns": [
      "Customer_Age",
      "Customer_Gender",
      "Customer_Segment",
      "Loyalty_Member",
      "Year",
      "Month",
      "Day",
      "Day_of_Week",
      "Weekend",
      "Hour",
      "Season",
      "Festival",
      "Weather",
      "Temperature",
      "Product",
      "Category",
      "Quantity",
      "Unit_Price",
      "Cost_Price",
      "Discount_Percentage",
      "Selling_Price",
      "Total_Bill",
      "Payment_Method",
      "Promotion_Applied",
      "Promotion_Type",
      "Shelf_Life_Days",
      "Storage_Type",
      "Perishability",
      "Store_ID"
    ],
    "data": [
      [
        61,
        "Male",
        "Senior Citizen",
        "Yes",
        2023,
        9,
        1,
        "Friday",
        "No",
        15,
        "Monsoon",
        null,
        "Rainy",
        26.6,
        "Soft Drink",
        "Beverages",
        1,
        44.47,
        24.66,
        27.8,
        32.11,
        32.11,
        "Wallet",
        "Yes",
        "Clearance",
        257,
        "Dry Storage",
        "Low",
        "ST03"
      ]
    ]
  }
}
```

## Respuesta

```json
{
  "predictions": [
    1
  ]
}
```
