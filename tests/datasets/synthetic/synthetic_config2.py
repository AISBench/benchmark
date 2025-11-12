synthetic_config = {
    "Type":"string",
    "RequestCount": 5,
    "StringConfig" : {
        "Input" : {
            "Method": "uniform",
            "Params": {"MinValue": 130, "MaxValue": 130}
        },
        "Output" : {
            "Method": "gaussian",
            "Params": {"Mean": 100, "Var": 200, "MinValue": 200, "MaxValue": 200}
        }
    },
    "TokenIdConfig" : {
        "ModelPath": "",
        "RequestSize": 10
    }
}