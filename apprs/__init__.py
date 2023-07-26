def create_method(args):
    if args.method == 'ROW':
        from apprs.row import ROW as Model
    return Model(args)