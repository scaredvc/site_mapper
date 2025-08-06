import pandas as pd

def analyze_label_differences():
    """Analyze where simple and contextual labels differ"""
    
    # Load the data
    df = pd.read_csv('results/training_data_v2.csv')
    
    print(f"Total links: {len(df)}")
    print(f"Simple 'good' links: {df['label_simple'].sum()}")
    print(f"Contextual 'good' links: {df['label_contextual'].sum()}")
    print(f"Difference: {df['label_contextual'].sum() - df['label_simple'].sum()}")
    
    # Find where labels differ
    different_labels = df[df['label_simple'] != df['label_contextual']]
    print(f"\nLinks where labels differ: {len(different_labels)}")
    
    if len(different_labels) > 0:
        print("\n=== Cases where Contextual says GOOD but Simple says BAD ===")
        contextual_good = different_labels[
            (different_labels['label_contextual'] == True) & 
            (different_labels['label_simple'] == False)
        ]
        
        for idx, row in contextual_good.iterrows():
            print(f"\nURL: {row['url']}")
            print(f"Text: '{row['link_text']}'")
            print(f"Has pagination: {row['has_pagination']}")
            print(f"Is main list pagination: {row['is_main_list_pagination']}")
            print(f"Has show param: {row['has_show_param']}")
            print(f"Is view toggle: {row['is_view_toggle']}")
            print(f"In faceted search: {row['in_faceted_search']}")
            
        print(f"\nTotal contextual-good/simple-bad: {len(contextual_good)}")
        
        print("\n=== Cases where Simple says GOOD but Contextual says BAD ===")
        simple_good = different_labels[
            (different_labels['label_simple'] == True) & 
            (different_labels['label_contextual'] == False)
        ]
        
        for idx, row in simple_good.iterrows():
            print(f"\nURL: {row['url']}")
            print(f"Text: '{row['link_text']}'")
            print(f"Has pagination: {row['has_pagination']}")
            print(f"Has show param: {row['has_show_param']}")
            print(f"Potential issues: {row['potential_issues_count']}")
            
        print(f"\nTotal simple-good/contextual-bad: {len(simple_good)}")
    
    # Look at pagination specifically
    pagination_links = df[df['has_pagination'] == True]
    print(f"\n=== Pagination Analysis ===")
    print(f"Total pagination links: {len(pagination_links)}")
    print(f"Main list pagination: {pagination_links['is_main_list_pagination'].sum()}")
    print(f"Nested pagination: {pagination_links['is_nested_pagination'].sum()}")
    
    if len(pagination_links) > 0:
        print("\nSample pagination links:")
        for idx, row in pagination_links.head(3).iterrows():
            print(f"  {row['url']} - Simple: {row['label_simple']}, Contextual: {row['label_contextual']}")
    
    # Look at show parameters
    show_links = df[df['has_show_param'] == True]
    print(f"\n=== Show Parameter Analysis ===")
    print(f"Total show param links: {len(show_links)}")
    print(f"View toggles: {show_links['is_view_toggle'].sum()}")
    
    if len(show_links) > 0:
        print("\nSample show parameter links:")
        for idx, row in show_links.head(3).iterrows():
            print(f"  {row['url']} - Simple: {row['label_simple']}, Contextual: {row['label_contextual']}")
            print(f"    Show value: '{row['show_param_value']}', Is view toggle: {row['is_view_toggle']}")

if __name__ == "__main__":
    analyze_label_differences()